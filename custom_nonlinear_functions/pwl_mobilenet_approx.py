import torch
from custom_approx import CustomSilu

# Has no parameters, only needs to be ran once

class PWLMobilenet(CustomSilu):
    def __init__(self, device, path, save_dims, profile):
        super(PWLMobilenet, self).__init__(device, path, save_dims, profile)

    def nonlinear(ctx, x):
        return x * (torch.nn.functional.relu6(x + 3) / 6)