__author__ = "Marc Kaufed"
__copyright__ = "TUM Institute of Automotive Technology"
__version__ = "1.0"
__maintainer__ = "Marc Kaufeld"
__email__ = "marc.kaufeld@tum.de"
__status__ = "Beta"

import json
import torch
import ml_planner.sampling.networks.rbf_network as networks


def load_model(model_path, model_type, device):
    # load config
    config_file = model_path / "config.json"
    with open(config_file, "r") as json_file:
        model_kwargs = json.load(json_file)
    # instantiate model
    model_cls = getattr(networks, model_type)
    model = model_cls(**model_kwargs)
    # load weights
    model_weigths_file = [str(file) for file in model_path.glob("*best.pth") if file.is_file()][-1]
    model.load_state_dict(torch.load(model_weigths_file, weights_only=True, map_location=device))
    return model, model_kwargs
