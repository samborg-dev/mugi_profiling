import torch
import math
from custom_approx import CustomSoftmax
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit a to adjust degree center
# Edit degrees to adjust number of polynomial degrees

class TaylorSoftmax(CustomSoftmax):
    def __init__(self, degree_center, degrees, device, path, save_dims, profile, torch_nonlinear):
        super(TaylorSoftmax, self).__init__(device, path, save_dims, profile, torch_nonlinear)
        self.degree_center = degree_center
        self.degrees = degrees
        self.device = device

        self.build_taylor()

    def reset_taylor(self, degree_center, degrees):
        self.degree_center = degree_center
        self.degrees = degrees
        self.build_taylor()

    def build_taylor(self):
        prev_exp = torch.tensor(1).to(torch.float16)
        x_neg = torch.tensor(0).to(torch.float64)
        exp = 0
        while True:
            for i in range(self.degrees):
                intermediate = x_neg - self.degree_center
                intermediate = intermediate ** i
                intermediate = intermediate / math.factorial(i)
                exp += intermediate
            exp *= (torch.e ** (self.degree_center))
            exp = exp.to(torch.float16)
            if exp <= 0.001 or prev_exp < exp or torch.isinf(exp):
                break
            else:
                prev_exp = exp.clone()
                x_neg -= 0.25
        self.x_neg = x_neg.to(torch.bfloat16)

    def taylor_exp(self, x):
        exp = 0
        for i in range(self.degrees + 1):
            intermediate = x - self.degree_center
            intermediate = intermediate ** i
            intermediate = intermediate / math.factorial(i)
            exp += intermediate
        exp *= (torch.e ** (self.degree_center))
        exp = torch.where(x < self.x_neg, torch.tensor(0.0, dtype=torch.bfloat16), exp)
        return exp
    
    def nonlinear(self, attn_weights, dim = -1):
        attn_weights = attn_weights.to(torch.bfloat16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights_scaled = attn_weights - attn_weights_max

        attn_weights_exp = self.taylor_exp(attn_weights_scaled)

        attn_weights_sum_exp = torch.sum(attn_weights_exp, dim = dim, keepdim = True)
        softmax_output = attn_weights_exp / attn_weights_sum_exp
        return softmax_output