import torch
from custom_nonlinear.custom_approx import CustomSoftmax

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit exp_dim to adjust the LUT size
# Edit max exp to adjust the maximum exponent of the LUT
class VLPSoftmax(CustomSoftmax):
    def __init__(self, exp_dim, max_exp, min_exp, window_size, lut_build, layer, device, profile_path, profile_dims, blocks=None):
        super(VLPSoftmax, self).__init__(layer, device, profile_path, profile_dims, blocks)
        # Exponent dimension of virtual LUT and maximum exponent for setting LUT range
        self.exp_dim = exp_dim
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.window_size = window_size
        self.lut_build = lut_build
        self.build_lut()

        # Mantissa approximation
        self.mant_dim = 8

    def reset_lut(self, exp_dim, max_exp, min_exp, window_size, lut_build):
        self.exp_dim = exp_dim
        self.max_exp = max_exp
        self.min_exp = min_exp
        self.window_size = window_size
        self.lut_build = lut_build
        self.build_lut()

    def build_lut(self):
        # Mantissa dimension of virtual LUT
        mant_dim = 8
        # set exp range by selected exp (max_exp or min_exp)
        if self.lut_build == "max":
            self.min_exp = self.max_exp - (self.exp_dim - 1)
        else:
            self.max_exp = self.min_exp + (self.exp_dim - 1)

        # generate exponent values of LUT
        exp_values = torch.arange(self.exp_dim).reshape(self.exp_dim, 1)
        exp_table = exp_values.expand(self.exp_dim, mant_dim)
        exp_table = exp_table + self.min_exp

        # generate mantissa values of LUT
        mant_values = torch.arange(mant_dim)
        mant_table = ((mant_values.expand(self.exp_dim, mant_dim) / 8) + 1) * -1

        # combine exponent and mantissa values
        exp_table = exp_table.to(torch.int32)
        mant_table = mant_table.to(torch.float32)
        lookup_table = torch.ldexp(mant_table, exp_table)

        # apply exp to create LUT
        self.lut = torch.exp(lookup_table).to(torch.bfloat16).to(self.device)

    def window_softmax_approx(self, exp, mant):
        attn_shape = exp.shape

        exp = exp.view(-1, attn_shape[-1])
        mant = mant.view(-1, attn_shape[-1])

        attn_inter_shape = exp.shape

        # Pad tensors
        if exp.shape[0] % self.window_size != 0:
            padding = self.window_size - (exp.shape[0] % self.window_size)

            padding_shape = [0] * len(exp.shape) * 2
            padding_shape[-1] = padding

            if self.lut_build == "max":
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=-1000)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=-1000)
            else:
                exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=1000)
                mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=1000)
        else:
            exp = exp
            mant = mant

        exp = exp.view(self.window_size, exp.shape[0] // self.window_size, exp.shape[1])
        mant = mant.view(self.window_size, mant.shape[0] // self.window_size, mant.shape[1])

        # calculate min and max windows
        if self.lut_build == "max":
            max_exp_window = torch.max(exp, dim=0, keepdim=True)[0].expand_as(exp)
            max_exp_window = torch.where(max_exp_window > self.max_exp, self.max_exp, max_exp_window)
            min_exp_window = max_exp_window - (self.mant_dim - 1)
            min_exp_window = torch.where(min_exp_window < self.min_exp, self.min_exp, min_exp_window)
        else:
            min_exp_window = torch.min(exp, dim=0, keepdim=True)[0].expand_as(exp)
            min_exp_window = torch.where(min_exp_window < self.min_exp, self.min_exp, min_exp_window)
            max_exp_window = min_exp_window + (self.mant_dim - 1)
            max_exp_window = torch.where(max_exp_window > self.max_exp, self.max_exp, max_exp_window)

        # compare to max/min values
        if self.lut_build == "max":
            exp_window_max = torch.where(exp <= max_exp_window, exp, max_exp_window)
            exp_window = torch.where(exp_window_max >= min_exp_window, exp_window_max, torch.where(exp_window_max == -1000, exp_window_max, min_exp_window))
            
            mant_window_max = torch.where(exp <= max_exp_window, mant, self.mant_dim - 1)
            mant_window = torch.where(exp_window_max >= min_exp_window, mant_window_max, torch.where(exp_window_max == -1000, mant_window_max, 0))
            del mant_window_max, exp_window_max
        else:
            exp_window_min = torch.where(exp >= min_exp_window, exp, min_exp_window)
            exp_window = torch.where(exp_window_min <= max_exp_window, exp_window_min, torch.where(exp_window_min == 1000, exp_window_min, max_exp_window))
            
            mant_window_min = torch.where(exp >= min_exp_window, mant, 0)
            mant_window = torch.where(exp_window_min <= max_exp_window, mant_window_min, torch.where(exp_window_min == 1000, mant_window_min, self.mant_dim - 1))
            del mant_window_min, exp_window_min

        del max_exp_window, min_exp_window

        # adjust to index lut
        if self.lut_build == "max":
            exp = exp_window - self.min_exp
        else:
            exp = exp_window - self.max_exp

        mant = mant_window
        
        del exp_window, mant_window

        # reshape to original shape
        exp = exp.view(-1, attn_shape[-1])
        mant = mant.view(-1, attn_shape[-1])

        exp = exp[:attn_inter_shape[0], :]
        mant = mant[:attn_inter_shape[0], :]

        exp = exp.view(*attn_shape)
        mant = mant.view(*attn_shape)

        return exp, mant

    def nonlinear(self, attn_weights, dim=-1, dtype=torch.bfloat16):
        attn_weights = attn_weights.to(torch.bfloat16)
        # Find the max value and subtract it to prevent overflow (max value is 0)
        attn_weights_max = torch.max(attn_weights, dim = dim, keepdim = True)[0]
        attn_weights = attn_weights - attn_weights_max

        del attn_weights_max  # Free memory immediately

        # Split exponent and signed mantissa, bitshift mantissa to 4 bits (assumes leading 0).
        mant, exp = torch.frexp(attn_weights)
        mant = torch.round(mant * 16)

        # Increment exponent where mantissa has overflow (i.e., mantissa is 16 / needs)
        # exp = torch.where(attn_weights == 0, exp, exp - 1)
        exp[attn_weights != 0] -= 1
        # exp = torch.where(torch.abs(mant) == 16, exp + 1, exp)
        mant = torch.abs(mant.to(torch.int32))
        exp[mant == 16] += 1

        max_exp_mask = exp > self.max_exp

        # Convert mantissa to unsigned 3 bit integer
        mant = mant & 0x7
        
        # Remove inf values to increase window selection stability
        # exp_processed = torch.where((torch.isinf(attn_weights)) | (attn_weights <= -65500.0), 0, exp)
        exp[attn_weights <= -65500.0] = 0
        
        # Clamp exponent and adjust mantissa to fit within LUT range
        exp, mant = self.window_softmax_approx(exp, mant)

        # Postprocess 0 case and large exponent case
        # exponentials = torch.where(attn_weights == 0, 1, self.lut[exp_processed, mant])
        exponentials = self.lut[exp, mant]

        exponentials[attn_weights == 0] = 1
        # exponentials = torch.where(exp > self.max_exp, 0, exponentials)
        exponentials[max_exp_mask] = 0

        # Calculate softmax output
        attn_weights = torch.sum(exponentials, dim = dim, keepdim = True)
        attn_weights = exponentials / attn_weights

        return attn_weights