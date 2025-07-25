import torch
import inspect

class CustomSoftmax(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile):
        super(CustomSoftmax, self).__init__()
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_softmax = torch.nn.functional.softmax

    def forward(self, attn_weights, dim=-1, dtype=torch.float32):
        function = inspect.stack()[1].function
        if function == "eager_attention_forward":
            if self.profile_bool:
                self.profile(attn_weights, dim=dim)
            return self.nonlinear(attn_weights, dim=dim).to(attn_weights.dtype)
        else:
            return self.torch_softmax(attn_weights, dim=dim, dtype=dtype)
    
    def nonlinear(self, attn_weights, dim=-1):
        return self.torch_softmax(attn_weights, dim=dim)
    
    def profile(self, attn_weights, dim=-1):
        attn_weights.to(torch.float16)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights_shifted = attn_weights - attn_weights_max

        # for save_dim in self.save_dims:
        #     values = attn_weights_shifted[:, :, save_dim, :(save_dim+1)].flatten()

        #     mant, exp = torch.frexp(values)

        #     # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
        #     exp = torch.where(exp != 0, exp + 16, exp)
        #     exp_count = torch.bincount(exp, minlength=32)

        #     # bin values into 0.25 steps
        #     value_edges = torch.arange(0, -20.25, -0.25).flip(0).to(self.device)
        #     value_indices = torch.bucketize(values, value_edges, right=True)
        #     value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
        #     value_count = value_count[1:-1]

        print(attn_weights_shifted[0, 0, 512, 256:266])

            #torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            #torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")
    
class CustomSilu(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile):
        super(CustomSilu, self).__init__()
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_silu = torch.nn.functional.silu

    def forward(self, x):
        if self.profile_bool:
            self.profile(x)
        return self.nonlinear(x).to(x.dtype)
    
    def nonlinear(self, x):
        return self.torch_silu(x)
    
    def profile(self, x):
        x.to(torch.float16)

        for save_dim in self.save_dims:
            values = x[:, save_dim, :].flatten()

            mant, exp = torch.frexp(values)

            # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
            exp = torch.where(exp != 0, exp + 16, exp)
            exp_count = torch.bincount(exp, minlength=32)

            # bin values into 0.25 steps
            value_edges = torch.arange(20, -20.25, -0.25).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]

            torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")
    
class CustomGelu(torch.nn.Module):
    def __init__(self, device, path, save_dims, profile):
        super(CustomGelu, self).__init__()
        self.device = device
        self.path = path
        self.save_dims = save_dims
        self.profile_bool = profile
        self.torch_silu = torch.nn.functional.gelu

    def forward(self, x):
        if self.profile_bool:
            self.profile(x)
        return self.nonlinear(x)
    
    def nonlinear(self, x):
        return self.torch_silu(x)
    
    def profile(self, x):
        x.to(torch.float16)

        for save_dim in self.save_dims:
            values = x[:, save_dim, :].flatten()

            mant, exp = torch.frexp(values)

            # keep 0 exp values, add 16 to all other to shift (offset + 1 for frexp format)
            exp = torch.where(exp != 0, exp + 16, exp)
            exp_count = torch.bincount(exp, minlength=32)

            # bin values into 0.25 steps
            value_edges = torch.arange(20, -20.25, -0.25).flip(0).to(self.device)
            value_indices = torch.bucketize(values, value_edges, right=True)
            value_count = torch.bincount(value_indices, minlength=len(value_edges)+1)
            value_count = value_count[1:-1]

            torch.save(exp_count, f"{self.path}/exp_count_{save_dim}.pt")
            torch.save(value_count, f"{self.path}/value_count_{save_dim}.pt")