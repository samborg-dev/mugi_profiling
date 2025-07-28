import torch
from custom_approx import CustomSoftmax
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (softmax range is set from segment_0 to 0 ex. -20 = [-20, 0])

class PWLSoftmax(CustomSoftmax):
    def __init__(self, segments, segment_0, device, path, save_dims, profile, torch_nonlinear):
        super(PWLSoftmax, self).__init__(device, path, save_dims, profile, torch_nonlinear)
        self.segments = segments - 1
        self.segment_0 = segment_0
        self.segment_f = 0
        self.device = device

        self.build_lut()

    def reset_lut(self, segments, segment_0):
        self.segments = segments - 1
        self.segment_0 = -segment_0
        self.segment_f = 0
        self.build_lut()

    def build_lut(self):
        self.x_segments = torch.linspace(self.segment_0, self.segment_f, self.segments + 1, dtype = torch.bfloat16).to(self.device)
        self.y_segments = torch.exp(self.x_segments).to(self.device).to(torch.bfloat16)
        mb = [self.ymxb(self.y_segments[i], self.y_segments[i+1], self.x_segments[i], self.x_segments[i+1]) for i in range(0, self.segments)]
        self.m = torch.tensor([mb[i][0].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)
        self.b = torch.tensor([mb[i][1].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)

    def ymxb(self, y0, y1, x0, x1):
        m = (y1 - y0) / (x1 - x0)
        b = (y0 - m * x0)
        return m, b
    
    def nonlinear(self, attn_weights, dim=-1):
        attn_weights = attn_weights.to(torch.bfloat16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights_scaled = attn_weights - attn_weights_max

        mask = (attn_weights_scaled.unsqueeze(-1) >= self.x_segments[:-1])
        coeffs = mask * (self.m.unsqueeze(0) * attn_weights_scaled.unsqueeze(-1) + self.b.unsqueeze(0))
        attn_weights_exp = coeffs.max(dim=-1).values

        attn_weights_exp = torch.where(attn_weights_scaled < self.x_segments[0], 0, attn_weights_exp)
        attn_weights_exp = torch.where(attn_weights_scaled > self.x_segments[-1], self.m[-1] * attn_weights_scaled + self.b[-1], attn_weights_exp)
        attn_weights_sum_exp = torch.sum(attn_weights_exp, dim = dim, keepdim = True)
        softmax_output = attn_weights_exp / attn_weights_sum_exp

        return softmax_output