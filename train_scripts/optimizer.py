import torch

def init_optimizer(model,cfg):
    bert_params = []
    pooling_params = []

    for name, param in model.named_parameters():
        if "pooling_layer" in name:     # attention pooling stuff
            pooling_params.append(param)
        else:
            bert_params.append(param)
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": float(cfg["training"]["learning_rate"])},  # e.g. 2e-5
        {"params": pooling_params, "lr": float(cfg["training"]["pooling_learning_rate"])},  # e.g. 1e-3
    ])
    return optimizer
