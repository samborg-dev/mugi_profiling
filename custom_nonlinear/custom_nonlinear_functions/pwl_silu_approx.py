import torch
from custom_nonlinear.custom_approx import CustomSilu
import os

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (SiLU range is set on both sides ex. 4 = [-4, 4])

class PWLSilu(CustomSilu):
    def __init__(self, segments, segment_0, layer, device, profile_path, profile_dims, blocks=None, keys=None):
        super(PWLSilu, self).__init__(layer, device, profile_path, profile_dims, blocks, keys)
        self.segments = segments - 2
        self.segment_0 = -segment_0
        self.segment_f = segment_0
        self.device = device

        self.build_lut()

    def reset_lut(self, segments, segment_0):
        self.segments = segments - 2
        self.segment_0 = -segment_0
        self.segment_f = segment_0

        self.build_lut()

    def build_lut(self):
        self.x_segments = torch.linspace(self.segment_0, self.segment_f, self.segments + 1, dtype = torch.bfloat16).to(self.device)
        self.y_segments = torch.nn.functional.silu(self.x_segments).to(self.device).to(torch.bfloat16)
        mb = [self.ymxb(self.y_segments[i], self.y_segments[i+1], self.x_segments[i], self.x_segments[i+1]) for i in range(0, self.segments)]
        self.m = torch.tensor([mb[i][0].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)
        self.b = torch.tensor([mb[i][1].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)

    def ymxb(self, y0, y1, x0, x1):
        m = (y1 - y0) / (x1 - x0)
        b = (y0 - m * x0)
        return m, b
    
    def nonlinear(self, x):
        x = x.to(torch.bfloat16)

        segment_indices = torch.clamp(x, min=self.x_segments[0], max=self.x_segments[-2])
        segment_indices = torch.bucketize(segment_indices, self.x_segments, right=True)
        segment_indices -= 1
        segment_indices = torch.clamp(segment_indices, 0, self.segments - 1)
        
        m = self.m[segment_indices]
        b = self.b[segment_indices]
        silu_output = m * x + b

        # mask = (x.unsqueeze(-1) >= self.x_segments[:-1])
        # coeffs = mask * (self.m.unsqueeze(0) * x.unsqueeze(-1) + self.b.unsqueeze(0))
        # mask = mask.long().sum(dim=-1) - 1
        # mask = torch.clamp(mask, 0, len(self.m)-1)

        silu_output = torch.where(x < self.x_segments[0], 0, silu_output).to(torch.bfloat16)
        silu_output = torch.where(x >= self.x_segments[-1], x, silu_output).to(torch.bfloat16)

        return silu_output