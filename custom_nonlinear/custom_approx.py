import torch
import os
from transformers.activations import FastGELUActivation

class CustomNonlinear(torch.nn.Module):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None):
        super(CustomNonlinear, self).__init__()
        self.layer = layer
        self.device = device
        self.profile_path = profile_path
        self.profile_dims = profile_dims
        self.blocks = blocks
        if not os.path.exists(self.profile_path):
            os.makedirs(self.profile_path, exist_ok=True)

    def process_tensor(self, tensor):
        if tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        return tensor

    def index_tensor(self, tensor, dim, index):
        indexer = [slice(None)] * tensor.ndim
        indexer[dim] = index
        return tensor[tuple(indexer)]

    def profile(self, tensor, dim, profile_path, left_value_edge, right_value_edge, value_index):
        dim_len = tensor.shape[dim]
        if self.profile_dims == -1:
            self.profile_dims = [(dim_len - 1) // 4,
                                 (dim_len - 1) // 2,
                                 (dim_len - 1)]

        tensor = self.process_tensor(tensor)
        tensor_dim = tensor.shape[dim]
        break_loop = False
        for i, save_dim in enumerate(self.profile_dims):
            if tensor_dim < save_dim:
                save_dim = tensor_dim - 1
                write_dim = i
                break_loop = True
            else:
                write_dim = save_dim
            values = self.index_tensor(tensor, dim, save_dim).contiguous()

            mant, exp = torch.frexp(values)
            del mant

            exp = torch.where(exp != 0, exp + 128, exp).flatten()
            exp_count = torch.bincount(exp, minlength=256)
            del exp

            value_edges = torch.arange(right_value_edge, left_value_edge, value_index).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_indices = value_indices.flatten().long()
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]
            del values, value_edges, value_indices

            exp_path = f'{profile_path}/exp_dist/layer_{self.layer}/'
            if self.blocks is not None:
                exp_path = os.path.join(exp_path, f'block_{self.blocks}/')
            exp_path = os.path.join(self.profile_path, exp_path)
            exp_file = os.path.join(exp_path, f'seq_len_{write_dim}.pt')

            value_path = f'{profile_path}/value_dist/layer_{self.layer}/'
            if self.blocks is not None:
                value_path = os.path.join(value_path, f'block_{self.blocks}/')
            value_path = os.path.join(self.profile_path, value_path)
            value_file = os.path.join(value_path, f'seq_len_{write_dim}.pt')

            if os.path.exists(exp_file):
                prev_exp_count = torch.load(exp_file)
                if len(prev_exp_count) != len(exp_count):
                    raise ValueError(f"Previous exp count length {len(prev_exp_count)} does not match current {len(exp_count)} for save_dim {write_dim}.")
                exp_count += prev_exp_count
            else:
                os.makedirs(os.path.dirname(exp_file), exist_ok=True)
            torch.save(exp_count, exp_file)

            if os.path.exists(value_file):
                prev_value_count = torch.load(value_file)
                if len(prev_value_count) != len(value_count):
                    raise ValueError(f"Previous value count length {len(prev_value_count)} does not match current {len(value_count)} for save_dim {write_dim}.")
                value_count += prev_value_count
            else:
                os.makedirs(os.path.dirname(value_file), exist_ok=True)
            torch.save(value_count, value_file)

            if break_loop:
                break

class CustomSoftmax(CustomNonlinear):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None):
        profile_path += 'softmax/'
        super().__init__(layer, device, profile_path, profile_dims, blocks)

    def forward(self, attn_weights, dim=-1, dtype=torch.float32):
        self.profile(attn_weights,
                     dim=dim,
                     profile_path='pre_softmax',
                     left_value_edge=-20.25,
                     right_value_edge=20,
                     value_index=-0.05)
        attn_weights = self.nonlinear(attn_weights, dim=dim, dtype=dtype).to(attn_weights.dtype)
        self.profile(attn_weights,
                     dim=dim,
                     profile_path='post_softmax',
                     left_value_edge=-20.25,
                     right_value_edge=0,
                     value_index=-0.05)
        return attn_weights
    
    def nonlinear(self, attn_weights, dim=-1, dtype=torch.float32):
        return torch.nn.functional.softmax(attn_weights, dim=dim, dtype=dtype)

class CustomSilu(CustomNonlinear):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None):
        profile_path += 'silu/'
        super().__init__(layer, device, profile_path, profile_dims, blocks)

    def forward(self, x):
        self.profile(x,
                     dim=1,
                     profile_path='pre_silu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        x = self.nonlinear(x).to(x.dtype)
        self.profile(x,
                     dim=1,
                     profile_path='post_silu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        return x
    
    def nonlinear(self, x):
        return torch.nn.functional.silu(x)

class CustomGelu(CustomNonlinear):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None):
        profile_path += 'gelu/'
        super().__init__(layer, device, profile_path, profile_dims, blocks)

    def forward(self, x):
        self.profile(x,
                     dim=1,
                     profile_path='pre_gelu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        x = self.nonlinear(x).to(x.dtype)
        self.profile(x,
                     dim=1,
                     profile_path='post_gelu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        return x
    
    def nonlinear(self, x):
        return torch.nn.functional.gelu(x)
    
class CustomFastGelu(CustomNonlinear):
    def __init__(self, layer, device, profile_path, profile_dims, blocks=None):
        profile_path += 'gelu/'
        super().__init__(layer, device, profile_path, profile_dims, blocks)
        self.fast_gelu = FastGELUActivation()

    def forward(self, x):
        self.profile(x,
                     dim=1,
                     profile_path='pre_gelu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        x = self.nonlinear(x).to(x.dtype)
        self.profile(x,
                     dim=1,
                     profile_path='post_gelu',
                     left_value_edge=-10.25,
                     right_value_edge=10,
                     value_index=-0.05)
        return x
    
    def nonlinear(self, x):
        return self.fast_gelu.forward(x)