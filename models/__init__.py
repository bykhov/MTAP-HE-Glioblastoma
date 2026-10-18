from torch import nn
from .clam_mil import CLAMSBModel

def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Model factory.
    
    Returns an instance of a model based on its name.
    To add a new model, create a new file in this directory and import it here.
    """
    if model_name == "clam_sb":
        return CLAMSBModel(**kwargs)
    else:
        raise ValueError(f"Model '{model_name}' not recognized.")