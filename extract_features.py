import os
import torch
import h5py
import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import torch.nn as nn
import warnings
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

warnings.filterwarnings("ignore", ".*does not have many workers.*")

class FeatureExtractor(nn.Module):
    """UNI (ViT-L/16, 1024-d tile embeddings) from the Hugging Face hub.

    The weights are gated (MahmoodLab/uni): authenticate first with `huggingface-cli login`
    or the HF_TOKEN environment variable. Set HF_HUB_CACHE to load a local copy.
    """
    def __init__(self, name: str = "uni"):
        super().__init__()
        if name != "uni":
            raise ValueError(f"Unsupported feature extractor {name!r}; only 'uni' is implemented.")
        self.feature_extractor = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

class PatchDataset(Dataset):
    """A simple dataset to load patches for a single slide."""
    def __init__(self, image_paths: list, original_indices, transform=None):
        self.image_paths = image_paths
        self.original_indices = original_indices
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if idx >= len(self.image_paths):
            return None, None
        path = self.image_paths[idx]
        try:
            with Image.open(path).convert("RGB") as img:
                if self.transform:
                    img = self.transform(img)
                return img, self.original_indices[idx]
        except FileNotFoundError:
            print(f"Warning: Image not found at {path}. Skipping.")
            return None, None
        except Exception as e:
            print(f"Error loading image {path}: {e}. Skipping.")
            return None, None

def collate_patches(batch: list):
    """
    Collates a batch of (image, original_index) tuples, filtering out None entries.
    """
    batch = [(img, idx) for img, idx in batch if img is not None and idx is not None]
    if not batch:
        return None, None  # Return None for both images and indices if batch is empty

    images, indices = zip(*batch)  # Unzip images and indices
    return torch.stack(images), list(indices)

def load_extractor(cfg):
    """The feature extractor in eval mode on its device, and the input transform that goes with it."""
    print(f"Loading feature extractor: {cfg.feature_extractor.name}")
    device = torch.device("cuda" if cfg.accelerator in ["gpu", "auto"] and torch.cuda.is_available() else "cpu")
    model = FeatureExtractor(name=cfg.feature_extractor.name)
    model.to(device)
    model.eval()

    # UNI's own input transform (resize to 224 px, ImageNet mean and std)
    transform = create_transform(**resolve_data_config(model.feature_extractor.pretrained_cfg, model=model.feature_extractor))
    return model, transform, device


def extract_slide(cfg, slide_id, slide_df, model, transform, device, h5_dir):
    """
    Features of one slide, written to ``<h5_dir>/<slide_id>.h5`` (one dataset per patch id).
    Returns the slide-level manifest entry, or None if the slide has no readable patch.
    """
    expected_dim = cfg.feature_extractor.get("feature_dim")
    image_paths = slide_df[cfg.image_path_col].tolist()
    # Store original indices from the slide_df to map features back to patch_ids
    original_indices_in_slide_df = slide_df.index.tolist()

    dataset = PatchDataset(image_paths, original_indices_in_slide_df, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        shuffle=False,
        collate_fn=collate_patches,
        pin_memory=True,
        drop_last=False  # Ensure all patches are processed, even if batch is smaller
    )

    slide_patch_features = []  # To store (patch_id, feature_vector) for this slide

    with torch.no_grad():
        for batch_images, batch_indices in dataloader:
            if batch_images is None:  # No valid images or indices in this batch
                continue

            batch_images = batch_images.to(device)
            features = model(batch_images)  # Features for the batch
            if expected_dim is not None and features.shape[-1] != expected_dim:
                raise ValueError(
                    f"The {cfg.feature_extractor.name} extractor produced {features.shape[-1]}-d features, "
                    f"but feature_extractor.feature_dim is {expected_dim}. Check the extract config."
                )

            # Map features back to their patch_ids
            for i, feature_vector in enumerate(features):
                original_idx = batch_indices[i]
                patch_id = slide_df.loc[original_idx, cfg.patch_id_col]
                slide_patch_features.append(
                    (str(patch_id), feature_vector.cpu()))  # Store as string for H5 dataset name

    if not slide_patch_features:
        print(f"Warning: No valid patches found for slide {slide_id}. Skipping.")
        return None

    # Save features to H5
    h5_path = os.path.join(h5_dir, f"{slide_id}.h5")
    with h5py.File(h5_path, 'w') as f:
        for patch_id_str, feature_vector in slide_patch_features:
            # H5 dataset names must be strings.
            f.create_dataset(patch_id_str, data=feature_vector.numpy())

    # One manifest entry per slide; case_id and label are constant within a slide
    first_patch_row = slide_df.iloc[0]
    return {
        "case_id": first_patch_row.get("case_id", slide_id),
        "slide_id": slide_id,
        "reference_to_slide_h5": os.path.abspath(h5_path),
        "slide_label": first_patch_row[cfg.label_col]
    }


def extract_features(cfg):
    """Main feature extraction routine."""
    pl.seed_everything(cfg.seed)

    # --- Setup ---
    print("Setting up directories and loading data...")
    output_dir = cfg.output_dir
    h5_dir = os.path.join(output_dir, cfg.output_h5_dir)
    os.makedirs(h5_dir, exist_ok=True)

    df = pd.read_parquet(cfg.source_parquet)
    slides = df.groupby(cfg.slide_id_col)

    # --- Model and transforms ---
    model, transform, device = load_extractor(cfg)

    # --- Main Loop ---
    print("Starting feature extraction...")
    slide_manifest_data = []

    for slide_id, slide_df in tqdm(slides, desc="Processing Slides"):
        entry = extract_slide(cfg, slide_id, slide_df, model, transform, device, h5_dir)
        if entry is not None:
            slide_manifest_data.append(entry)

    # --- Save Manifest ---
    print("Saving final slide-level feature manifest...")
    manifest_df = pd.DataFrame(slide_manifest_data)
    output_parquet_path = os.path.join(output_dir, cfg.output_parquet_name)
    manifest_df.to_parquet(output_parquet_path, index=False)
    print(f"Feature extraction complete. Manifest saved to: {output_parquet_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="UNI feature extraction. Trailing key=value arguments override the config "
                    "(e.g. source_parquet=... output_dir=... batch_size=64).")
    parser.add_argument("--config", default="configs/extract_config.yaml")
    args, overrides = parser.parse_known_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    extract_features(config)