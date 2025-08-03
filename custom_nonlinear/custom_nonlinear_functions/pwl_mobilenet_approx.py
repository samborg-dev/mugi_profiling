import torch
from custom_nonlinear.custom_approx import CustomSilu

# Has no parameters, only needs to be ran once

class PWLMobilenet(CustomSilu):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None, keys=None):
        super(PWLMobilenet, self).__init__(layer, device, profile_path, profile_dims, blocks, keys)

    def nonlinear(ctx, x):
        return x * (torch.nn.functional.relu6(x + 3) / 6)