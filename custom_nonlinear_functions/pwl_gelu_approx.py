import torch
from custom_approx import CustomGelu

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit segments to adjust the number of piecewise linear segments
# Edit segment_0 to set range (GeLU range is set on both sides ex. 4 = [-4, 4])

class PWLGelu(CustomGelu):
    def __init__(self, segments, segment_0, device, path, save_dims, profile):
        super(PWLGelu, self).__init__(device, path, save_dims, profile)
        self.segments = segments - 2
        self.segment_0 = -segment_0
        self.segment_f = segment_0
        self.device = device

        self.build_lut()

    def build_lut(self):
        self.x_segments = torch.linspace(self.segment_0, self.segment_f, self.segments + 1, dtype = torch.bfloat16).to(self.device)
        self.y_segments = torch.nn.functional.gelu(self.x_segments).to(self.device).to(torch.bfloat16)
        mb = [self.ymxb(self.y_segments[i], self.y_segments[i+1], self.x_segments[i], self.x_segments[i+1]) for i in range(0, self.segments)]
        self.m = torch.tensor([mb[i][0].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)
        self.b = torch.tensor([mb[i][1].item() for i in range(0, self.segments)], dtype=torch.bfloat16).to(self.device)

    def ymxb(self, y0, y1, x0, x1):
        m = (y1 - y0) / (x1 - x0)
        b = (y0 - m * x0)
        return m, b
    
    def nonlinear(self, x):
        x = x.to(torch.bfloat16)

        mask = (x.unsqueeze(-1) >= self.x_segments[:-1])
        coeffs = mask * (self.m.unsqueeze(0) * x.unsqueeze(-1) + self.b.unsqueeze(0))
        segment_indices = mask.long().sum(dim=-1) - 1
        segment_indices = torch.clamp(segment_indices, 0, len(self.m)-1)

        gelu_output = torch.gather(coeffs, -1, segment_indices.unsqueeze(-1)).squeeze(-1).to(torch.bfloat16)
        gelu_output = torch.where(x < self.x_segments[0], 0, gelu_output).to(torch.bfloat16)
        gelu_output = torch.where(x >= self.x_segments[-1], x, gelu_output).to(torch.bfloat16)

        return gelu_output