import torch
import inspect
import os

class CustomSoftmax(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile, torch_nonlinear):
        super(CustomSoftmax, self).__init__()
        if 'null' not in path:
            path = os.path.normpath(path)
            path = os.path.join(path, 'softmax')
            if os.path.exists(path) is False:
                os.makedirs(path, exist_ok=True)
        else:
            path = 'null'
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_nonlinear = torch_nonlinear

    def forward(self, attn_weights, dim=-1, dtype=torch.float32):
        if self.profile_bool:
            self.profile(attn_weights, dim=dim, type='pre_softmax')
        attn_weights = self.nonlinear(attn_weights, dim=dim).to(attn_weights.dtype)
        if self.profile_bool:
            self.profile(attn_weights, dim=dim, type='post_softmax')
        return attn_weights
    
    def nonlinear(self, attn_weights, dim=-1):
        return self.torch_nonlinear(attn_weights, dim=dim)
    
    def profile(self, attn_weights, dim=-1, type='pre_softmax'):
        if self.path == 'null':
            return
        
        attn_weights = attn_weights.to(torch.bfloat16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights = attn_weights - attn_weights_max
        del attn_weights_max  # Free memory immediately
        
        finish_loop = False
        for save_dim in self.save_dims:
            if attn_weights.shape[2] < save_dim + 1:
                save_dim = attn_weights.shape[2] - 1
                finish_loop = True

            values = attn_weights[:, :, save_dim, :(save_dim+1)].flatten()

            mant, exp = torch.frexp(values)
            del mant  # We don't use mant, so delete it

            # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
            exp = torch.where(exp != 0, exp + 128, exp).flatten()
            exp_count = torch.bincount(exp, minlength=256)
            del exp  # Free exp after using it

            # bin values into 0.25 steps
            value_edges = torch.arange(0, -20.25, -0.25).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]
            torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")
            del values, value_edges, value_indices, value_count, exp_count
            if finish_loop:
                break

class CustomSilu(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile, torch_nonlinear):
        super(CustomSilu, self).__init__()
        if 'null' not in path:
            path = os.path.normpath(path)
            path = os.path.join(path, 'softmax')
            if os.path.exists(path) is False:
                os.makedirs(path, exist_ok=True)
        else:
            path = 'null'
        if os.path.exists(path) is False:
            os.makedirs(path, exist_ok=True)
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_nonlinear = torch_nonlinear

    def forward(self, x):
        if self.profile_bool:
            self.profile(x)
        x = self.nonlinear(x).to(x.dtype)
        if self.profile_bool:
            self.profile(x)
        return x
    
    def nonlinear(self, x):
        return self.torch_nonlinear(x)
    
    def profile(self, x):
        if self.path == 'null':
            return
        
        x = x.to(torch.bfloat16)

        finish_loop = False
        for save_dim in self.save_dims:
            if x.shape[1] < save_dim + 1:
                save_dim = x.shape[1] - 1
                finish_loop = True
                
            values = x[:, save_dim, :].flatten()

            mant, exp = torch.frexp(values)
            del mant  # We don't use mant, so delete it

            # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
            exp = torch.where(exp != 0, exp + 128, exp).flatten()
            exp_count = torch.bincount(exp, minlength=256)
            del exp  # Free exp after using it

            # bin values into 0.25 steps
            value_edges = torch.arange(20, -20.25, -0.25).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]

            torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")
            del values, value_edges, value_indices, value_count, exp_count
            
            if finish_loop:
                break
    
class CustomGelu(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile, torch_nonlinear):
        super(CustomGelu, self).__init__()
        if 'null' not in path:
            path = os.path.normpath(path)
            path = os.path.join(path, 'softmax')
            if os.path.exists(path) is False:
                os.makedirs(path, exist_ok=True)
        else:
            path = 'null'
        if os.path.exists(path) is False:
            os.makedirs(path, exist_ok=True)
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_nonlinear = torch_nonlinear

    def forward(self, x):
        if self.profile_bool:
            self.profile(x)
        x = self.nonlinear(x).to(x.dtype)
        if self.profile_bool:
            self.profile(x)
        return x
    
    def nonlinear(self, x):
        return self.torch_nonlinear(x)
    
    def profile(self, x):
        if self.path == 'null':
            return

        x = x.to(torch.bfloat16)

        finish_loop = False
        for save_dim in self.save_dims:
            if x.shape[1] < save_dim + 1:
                save_dim = x.shape[1] - 1
                finish_loop = True
                
            values = x[:, save_dim, :].flatten()

            mant, exp = torch.frexp(values)
            del mant  # We don't use mant, so delete it

            # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
            exp = torch.where(exp != 0, exp + 128, exp).flatten()
            exp_count = torch.bincount(exp, minlength=256)
            del exp  # Free exp after using it

            # bin values into 0.25 steps
            value_edges = torch.arange(20, -20.25, -0.25).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]

            torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")
            del values, value_edges, value_indices, value_count, exp_count
            
            if finish_loop:
                break