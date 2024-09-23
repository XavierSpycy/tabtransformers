import yaml
from typing import Dict, Any

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

from .models.feature_tokenizer_transformer import FeatureTokenizerTransformer
from .models.tabular_transformer import TabularTransformer
from .metrics import root_mean_squared_logarithmic_error, f1_score_macro

def load_config(config_path: str):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def get_model(config: Dict[str, Any], vocabulary: Dict,  num_continuous_features: int):
    if not isinstance((model_class := config.get("model_class")), str):
        raise ValueError("model must be a string")

    if (model_kwargs := config.get("model_kwargs")) is None:
        model_kwargs = {}
    elif isinstance(model_kwargs, dict):
        model_kwargs = model_kwargs
    else:
        raise ValueError("model_kwargs must be a dictionary or None")

    if model_class == "tab_transformer":
        return TabularTransformer(vocabulary=vocabulary, num_continuous_features=num_continuous_features, **model_kwargs)
    elif model_class == "feature_tokenizer_transformer":
        return FeatureTokenizerTransformer(vocabulary=vocabulary, num_continuous_features=num_continuous_features, **model_kwargs)
    else:
        raise ValueError("model must be supported by tabtransformers")
    
def get_loss_function(config: Dict[str, Any]):
    if not isinstance((loss_function_name := config.get("loss_function")), str):
            raise ValueError("loss_function must be a string")

    if (loss_kwargs := config.get("loss_kwargs")) is None:
        loss_kwargs = {}
    elif isinstance(loss_kwargs, dict):
        loss_kwargs = loss_kwargs
    else:
        raise ValueError("loss_kwargs must be a dictionary or None")

    if loss_function_name == "l1":
        return nn.L1Loss(**loss_kwargs)
    elif loss_function_name == "mse":  
        return nn.MSELoss(**loss_kwargs)
    elif loss_function_name == "cross_entropy":
        return nn.CrossEntropyLoss(**loss_kwargs)
    elif loss_function_name == "ctc":
        return nn.CTCLoss(**loss_kwargs)
    elif loss_function_name == "nll":
        return nn.NLLLoss(**loss_kwargs)
    elif loss_function_name == "poisson_nll":
        return nn.PoissonNLLLoss(**loss_kwargs)
    elif loss_function_name == "gaussian_nll":
        return nn.GaussianNLLLoss(**loss_kwargs)
    elif loss_function_name == "kl_div":
        return nn.KLDivLoss(**loss_kwargs)
    elif loss_function_name == "bce":
        return nn.BCELoss(**loss_kwargs)
    elif loss_function_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(**loss_kwargs)
    elif loss_function_name == "margin_ranking":
        return nn.MarginRankingLoss(**loss_kwargs)
    elif loss_function_name == "hinge_embedding":
        return nn.HingeEmbeddingLoss(**loss_kwargs)
    elif loss_function_name == "multi_label_margin":
        return nn.MultiMarginLoss(**loss_kwargs)
    elif loss_function_name == "huber":
        return nn.HuberLoss(**loss_kwargs)
    elif loss_function_name == "smooth_l1":
        return nn.SmoothL1Loss(**loss_kwargs)
    elif loss_function_name == "soft_margin":
        return nn.SoftMarginLoss(**loss_kwargs)
    elif loss_function_name == "multi_label_soft_margin":
        return nn.MultiLabelSoftMarginLoss(**loss_kwargs)
    elif loss_function_name == "cosine_embedding":
        return nn.CosineEmbeddingLoss(**loss_kwargs)
    elif loss_function_name == "multi_margin":
        return nn.MultiMarginLoss(**loss_kwargs)
    elif loss_function_name == "triplet_margin":
        return nn.TripletMarginLoss(**loss_kwargs)
    elif loss_function_name == "triplet_margin_with_distance":
        return nn.TripletMarginWithDistanceLoss(**loss_kwargs)
    else:
        raise ValueError("loss_function must be supported by PyTorch")

def get_optimizer_object(config: Dict[str, Any]):
    if not isinstance((optimizer_name := config.get("optim")), str):
        raise ValueError("optimizer must be a string")

    if (optimizer_kwargs := config.get("optim_kwargs")) is None:
        optimizer_kwargs = {}
    elif isinstance(optimizer_kwargs, dict):
        optimizer_kwargs = optimizer_kwargs
    else:
        raise ValueError("optimizer_kwargs must be a dictionary or None")

    if optimizer_name == "adadelta":
        return optim.Adadelta
    elif optimizer_name == "adagrad":
        return optim.Adagrad
    elif optimizer_name == "adam":
        return optim.Adam
    elif optimizer_name == "adamw":
        return optim.AdamW
    elif optimizer_name == "sparse_adam":
        return optim.SparseAdam
    elif optimizer_name == "adamax":
        return optim.Adamax
    elif optimizer_name == "asgd":
        return optim.ASGD
    elif optimizer_name == "lbfgs":
        return optim.LBFGS
    elif optimizer_name == "nadam":
        return optim.Nadam
    elif optimizer_name == "radam":
        return optim.RAdam
    elif optimizer_name == "rmsprop":
        return optim.RMSprop
    elif optimizer_name == "rprop":
        return optim.Rprop
    elif optimizer_name == "sgd":
        return optim.SGD
    else:
        raise ValueError("optimizer must be supported by PyTorch")
    
def get_lr_scheduler_object(config: Dict[str, Any]):
    if (scheduler_name := config.get("lr_scheduler")) is None:
        return None
    elif not isinstance(scheduler_name, str):
        raise ValueError("scheduler must be a string")
    
    if scheduler_name == "step":
        return scheduler.StepLR
    elif scheduler_name == "multi_step":
        return scheduler.MultiStepLR
    elif scheduler_name == "constant":
        return scheduler.ConstantLR
    elif scheduler_name == "linear":
        return scheduler.LinerLR
    elif scheduler_name == "exponential":
        return scheduler.ExponentialLR
    elif scheduler_name == "polynomial":
        return scheduler.PolynomialLR
    elif scheduler_name == "cosine_annealing":
        return scheduler.CosineAnnealingLR
    elif scheduler_name == "reduce_on_plateau":
        return scheduler.ReduceLROnPlateau
    elif scheduler_name == "cyclic":
        return scheduler.CyclicLR
    elif scheduler_name == "one_cycle":
        return scheduler.OneCycleLR
    elif scheduler_name == "cosine_warm_restarts":
        return scheduler.CosineWarmRestarts
    else:
        raise ValueError("scheduler must be supported by PyTorch")

def get_custom_metric(config: Dict[str, Any]):
    if (custom_metric_name := config.get("custom_metric")) is None:
        return None
    elif not isinstance(custom_metric_name, str):
        raise ValueError("custom_metric must be a string")
    
    if custom_metric_name == "root_mean_squared_logarithmic_error":
        return root_mean_squared_logarithmic_error
    elif custom_metric_name == "f1_score_macro":
        return f1_score_macro
    else:
        raise ValueError("custom_metric must be supported by tabtransformers")