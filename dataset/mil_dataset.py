import pandas as pd
import torch
import pytorch_lightning as pl
import h5py
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import List, Tuple, Dict, Any, Optional
import warnings
import numpy as np

class MILDataset(Dataset):
    """
    Multi-Instance Learning Dataset.
    Each item is a "bag" of instances (patches) belonging to a single PATIENT.
    This dataset is optimized to read from patient-level H5 files containing
    'features' and 'patch_ids' datasets.
    """

    def __init__(self, df: pd.DataFrame, data_cfg: Dict[str, Any], is_train: bool = False):
        """
        Args:
            df (pd.DataFrame): DataFrame containing patient-level metadata.
            data_cfg (Dict): The data configuration dictionary.
            is_train (bool): Flag to indicate if this is the training dataset.
        """
        self.df = df.reset_index(drop=True)
        self.data_cfg = data_cfg
        self.is_train = is_train
        self.patient_id_col = self.data_cfg['case_id_col']
        self.label_col = self.data_cfg['label_col']
        self.h5_path_col = self.data_cfg['feature_h5_path_col']
        self.max_bag_size = self.data_cfg.get('max_bag_size_for_train', None)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor, str, bool, Optional[int]]:
        """
        Returns the path to the H5 file, the label, the patient ID, the is_train flag, and max_bag_size.
        The actual data reading is deferred to the collate function.
        """
        patient_info = self.df.iloc[index]
        patient_id = patient_info[self.patient_id_col]
        h5_path = patient_info[self.h5_path_col]
        label = torch.tensor(patient_info[self.label_col], dtype=torch.long)

        return h5_path, label, patient_id, self.is_train, self.max_bag_size


def collate_bags(batch: List[Tuple[str, torch.Tensor, str, bool, Optional[int]]]) -> tuple[
    Tensor, Tensor, Tensor, list[Any], list[Any]]:
    """
    Custom collate function optimized for I/O and memory efficiency.
    It opens each H5 file only once, inspects its metadata, pre-allocates tensors,
    and then loads data directly into memory using h5py's read_direct.
    During training, it can randomly subsample instances if a bag is too large.
    """
    h5_paths, labels, bag_ids, is_train_flags, max_bag_sizes = zip(*batch)
    is_train = is_train_flags[0]
    max_bag_size_for_train = max_bag_sizes[0] if is_train else None

    valid_samples = []
    open_files = []

    try:
        # First pass: Open files and gather metadata
        for i, h5_path in enumerate(h5_paths):
            try:
                f = h5py.File(h5_path, 'r')
                open_files.append(f)

                features_dset = f['features']
                n_instances = features_dset.shape[0]

                if n_instances == 0:
                    warnings.warn(f"No features found for bag {bag_ids[i]}. Skipping.")
                    continue

                patch_ids_bytes = f['patch_ids'][:]
                patch_ids = [pid.decode('utf-8') for pid in patch_ids_bytes]

                # --- Subsampling logic for training ---
                indices_to_load = np.arange(n_instances)
                if is_train and max_bag_size_for_train and n_instances > max_bag_size_for_train:
                    indices_to_load = np.random.choice(n_instances, max_bag_size_for_train, replace=False)
                    indices_to_load.sort()  # Sort indices for h5py read_direct requirement
                    n_instances = max_bag_size_for_train
                    patch_ids = [patch_ids[i] for i in indices_to_load]

                valid_samples.append({
                    "file_handle": f,
                    "features_dset": features_dset,
                    "n_instances": n_instances,
                    "indices_to_load": indices_to_load,
                    "patch_ids": patch_ids,
                    "label": labels[i],
                    "bag_id": bag_ids[i]
                })

            except Exception as e:
                warnings.warn(f"Could not read metadata for bag {bag_ids[i]} from {h5_path}. Error: {e}. Skipping.")
                continue

        if not valid_samples:
            return torch.empty(0), torch.empty(0), torch.empty(0), [], []

        # Determine batch dimensions
        max_len = max(s['n_instances'] for s in valid_samples)
        n_features = valid_samples[0]['features_dset'].shape[1] if valid_samples else 0
        batch_size = len(valid_samples)

        # Pre-allocate tensors
        padded_instances = torch.zeros((batch_size, max_len, n_features), dtype=torch.float32)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        final_labels = torch.stack([s['label'] for s in valid_samples])
        final_bag_ids = [s['bag_id'] for s in valid_samples]
        padded_patch_ids = []

        # Second pass: Load data directly into pre-allocated tensors from open files
        for i, sample in enumerate(valid_samples):
            length = sample['n_instances']

            try:
                # Use read_direct with the selected indices
                sample['features_dset'].read_direct(padded_instances.numpy(),
                                                    source_sel=np.s_[sample['indices_to_load'], :],
                                                    dest_sel=np.s_[i, :length, :])

                mask[i, :length] = True
                padded_patch_ids.append(sample['patch_ids'] + [None] * (max_len - length))

            except Exception as e:
                warnings.warn(
                    f"Error loading data for bag {sample['bag_id']}. Error: {e}. This bag will be zero-padded.")
                padded_patch_ids.append([None] * max_len)

        return padded_instances, final_labels, mask, final_bag_ids, padded_patch_ids

    finally:
        # Ensure all opened files are closed
        for f in open_files:
            try:
                f.close()
            except Exception as e:
                warnings.warn(f"Failed to close file handle. Error: {e}")


def patch_buckets(df: pd.DataFrame) -> pd.Series:
    """
    1 if a patient has more patches than the cohort median, else 0.

    The median is computed over the whole manifest passed in (the full cohort; 3,080 patches for
    the 233-patient cohort).
    """
    median_patches = float(np.median(df['n_patches']))
    return (df['n_patches'] > median_patches).astype(int)


def make_fold_indices(df: pd.DataFrame, label_col: str, k_folds: int, seed: int) -> List[np.ndarray]:
    """
    Positional indices of the k stratified folds (stratified by label and patch-count bucket).

    ``fold_indices[i]`` are the rows that belong to fold ``i``. In the run with ``fold_index = r``,
    fold ``r`` is the validation set, fold ``(r + 1) % k`` is the test set and the rest train.
    """
    if k_folds < 3:
        raise ValueError("K-fold cross-validation with a separate test set requires at least 3 folds.")

    stratify_col = df[label_col].astype(str) + '_' + patch_buckets(df).astype(str)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
    return [test_idx for _, test_idx in skf.split(df, stratify_col)]


def folds_table(df: pd.DataFrame, patient_id_col: str, label_col: str, k_folds: int, seed: int) -> pd.DataFrame:
    """
    One row per patient: ``patient_id, fold, label, n_patches``.

    ``fold`` is the run index ``r`` in which the patient is in the TEST set, so it matches the
    ``fold`` column of the ``fold{r}_test_predictions.csv`` files. (The same patient is in the
    validation set of run ``(r + 1) % k``.)
    """
    fold_indices = make_fold_indices(df, label_col, k_folds, seed)
    run_of_row = np.full(len(df), -1, dtype=int)
    for fold_number, rows in enumerate(fold_indices):
        run_of_row[rows] = (fold_number - 1) % k_folds
    return pd.DataFrame({
        'patient_id': df[patient_id_col].values,
        'fold': run_of_row,
        'label': df[label_col].values,
        'n_patches': df['n_patches'].values,
    })


def fold_indices_from_csv(df: pd.DataFrame, patient_id_col: str, label_col: str, k_folds: int,
                          folds_csv) -> List[np.ndarray]:
    """
    Fold membership read from a ``folds.csv`` (``patient_id, fold, label, n_patches``; ``fold`` is the run in
    which the patient is tested) instead of being recomputed from the manifest. Returns the same structure as
    ``make_fold_indices`` (``result[i]`` = rows of fold ``i``; the test fold of run ``r`` is ``(r + 1) % k``).

    The patients and labels in the file must equal those of the manifest; ``n_patches`` may differ, so a manifest
    with slightly different tile counts keeps the same split.
    """
    table = pd.read_csv(folds_csv)
    missing = {'patient_id', 'fold', 'label'} - set(table.columns)
    if missing:
        raise ValueError(f"{folds_csv}: missing column(s) {sorted(missing)}")
    if not table['patient_id'].is_unique or set(table['patient_id']) != set(df[patient_id_col]):
        raise ValueError(f"{folds_csv}: the patients differ from the manifest "
                         f"({len(set(table['patient_id']) ^ set(df[patient_id_col]))} not in both)")
    labels = table.set_index('patient_id')['label'].reindex(df[patient_id_col]).to_numpy()
    if not (labels == df[label_col].to_numpy()).all():
        raise ValueError(f"{folds_csv}: labels differ from the manifest")
    runs = table.set_index('patient_id')['fold'].reindex(df[patient_id_col]).to_numpy()
    if set(runs) != set(range(k_folds)):
        raise ValueError(f"{folds_csv}: expected fold values 0..{k_folds - 1}, found {sorted(set(runs))}")
    fold_number = (runs + 1) % k_folds  # the run in which a patient is tested -> its fold number
    return [np.where(fold_number == i)[0] for i in range(k_folds)]


def expected_input_dim(cfg: Dict[str, Any]) -> Optional[int]:
    """The feature dimension the model expects, or None if the config does not say."""
    model_cfg = cfg.get('model') or {}
    params = model_cfg.get('params')
    if params is None and model_cfg.get('name') is not None:
        params = model_cfg.get(model_cfg['name'])
    if params is None:
        return None
    return params.get('input_dim')


def check_feature_dim(df: pd.DataFrame, h5_path_col: str, expected_dim: Optional[int]) -> None:
    """
    Raise if the stored features do not have ``expected_dim`` columns. The first H5 file that
    can be opened is inspected; files that cannot be opened are left to ``collate_bags``.
    """
    if expected_dim is None:
        return
    for h5_path in df[h5_path_col]:
        try:
            with h5py.File(h5_path, 'r') as f:
                found_dim = f['features'].shape[1]
        except Exception:
            continue
        if found_dim != expected_dim:
            raise ValueError(
                f"Feature dimension mismatch: {h5_path} stores {found_dim}-d features but "
                f"model.params.input_dim is {expected_dim}. Re-extract the features with the model the "
                f"config expects (UNI is 1024-d) or set input_dim."
            )
        return
    warnings.warn("Could not open any patient H5 file to check the feature dimension.")


class MILDataModule(pl.LightningDataModule):
    """
    LightningDataModule for MIL.
    This version is optimized for patient-level H5 files.
    """

    def __init__(self, cfg: Dict[str, Any], fold_index: Optional[int] = None):
        super().__init__()
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.predict_dataset = None
        self.cfg = cfg
        self.fold_index = fold_index
        self.save_hyperparameters(cfg)

    def setup(self, stage: str = None):
        # One row per patient in the manifest
        full_df = pd.read_parquet(self.cfg['data']['parquet_path'])
        patient_id_col = self.cfg['data']['case_id_col']
        label_col = self.cfg['data']['label_col']

        # --- Stratification by patch count ---
        # patch_bucket splits the cohort at its median patch count
        full_df['patch_bucket'] = patch_buckets(full_df)

        check_feature_dim(full_df, self.cfg['data']['feature_h5_path_col'], expected_input_dim(self.cfg))

        if self.fold_index is not None:
            k_folds = self.cfg['data'].get('k_folds', 5)

            # Stratified by label and patch bucket; one list of row indices per fold
            folds_csv = self.cfg['data'].get('folds_csv')
            if folds_csv:
                fold_indices = fold_indices_from_csv(full_df, patient_id_col, label_col, k_folds, folds_csv)
            else:
                fold_indices = make_fold_indices(full_df, label_col, k_folds, self.cfg['seed'])

            # Assign folds for val, test, and train
            val_fold_idx = self.fold_index
            test_fold_idx = (self.fold_index + 1) % k_folds

            val_idx = fold_indices[val_fold_idx]
            test_idx = fold_indices[test_fold_idx]

            # The rest of the folds are for training
            train_fold_indices_nums = [i for i in range(k_folds) if i != val_fold_idx and i != test_fold_idx]
            train_idx = np.concatenate([fold_indices[i] for i in train_fold_indices_nums])

            train_df = full_df.iloc[train_idx]
            val_df = full_df.iloc[val_idx]
            test_df = full_df.iloc[test_idx]

            # Print average patch number for the fold
            print(f"--- Fold {self.fold_index + 1}/{k_folds} ---")
            print(f"Train set: {len(train_df)} patients (avg patches: {train_df['n_patches'].mean():.2f})")
            print(f"Val set:   {len(val_df)} patients (avg patches: {val_df['n_patches'].mean():.2f}) - Fold {val_fold_idx}")
            print(f"Test set:  {len(test_df)} patients (avg patches: {test_df['n_patches'].mean():.2f}) - Fold {test_fold_idx}")

        else:
            train_val_df, test_df = train_test_split(
                full_df,
                test_size=self.cfg['data']['test_split'],
                random_state=self.cfg['seed'],
                stratify=full_df[label_col]
            )
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.cfg['data']['val_split'] / (1 - self.cfg['data']['test_split']),
                random_state=self.cfg['seed'],
                stratify=train_val_df[label_col]
            )

            # Print average patch number for the splits
            print("--- Train/Val/Test Split ---")
            print(f"Train avg patches: {train_df['n_patches'].mean():.2f}")
            print(f"Val avg patches: {val_df['n_patches'].mean():.2f}")
            print(f"Test avg patches: {test_df['n_patches'].mean():.2f}")

        self.train_dataset = MILDataset(train_df, self.cfg['data'], is_train=True)
        self.val_dataset = MILDataset(val_df, self.cfg['data'])
        self.test_dataset = MILDataset(test_df, self.cfg['data'])
        self.predict_dataset = self.test_dataset

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        # If it's the training dataset and we want to use weighted sampling
        if isinstance(dataset, MILDataset) and dataset.is_train and self.cfg['datamodule'].get('use_weighted_sampler', False):
            print("\n--- Activating WeightedRandomSampler for training ---")
            # 1. Get labels from the training dataset
            labels = dataset.df[self.cfg['data']['label_col']].values

            # 2. Calculate class weights
            class_counts = np.bincount(labels)
            print(f"Class counts (0 and 1): {class_counts}")
            # Handle potential division by zero if a class is missing
            class_weights = 1. / np.where(class_counts == 0, 1, class_counts)
            print(f"Calculated class weights: {class_weights}")
            
            # 3. Assign a weight to each sample based on its label
            sample_weights = class_weights[labels]

            # 4. Create the WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights), # Draw 'num_samples' elements
                replacement=True # Usually True for weighted sampling
            )
            
            # When using a sampler, shuffle should be False
            return DataLoader(
                dataset,
                batch_size=self.cfg['datamodule']['batch_size'],
                num_workers=self.cfg['datamodule']['num_workers'],
                collate_fn=collate_bags,
                sampler=sampler, # Use the sampler here
                shuffle=False, # Set shuffle to False when using a sampler
                pin_memory=True,
                persistent_workers=self.cfg['datamodule']['persistent_workers'] and self.cfg['datamodule']['num_workers'] > 0
            )
        else:
            # For validation, test, or if weighted sampling is not enabled
            return DataLoader(
                dataset,
                batch_size=self.cfg['datamodule']['batch_size'],
                num_workers=self.cfg['datamodule']['num_workers'],
                collate_fn=collate_bags,
                shuffle=shuffle,
                pin_memory=True,
                persistent_workers=self.cfg['datamodule']['persistent_workers'] and self.cfg['datamodule']['num_workers'] > 0
            )

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.predict_dataset, shuffle=False)
