import torch
from custom_nonlinear.custom_approx import CustomSilu

# Code functions with fp16 precision as input / output, and is not tested for other datatypes.
# Edit exp_dim to adjust the LUT size
# Edit max exp to adjust the maximum exponent of the LUT
class VLPSilu(CustomSilu):
    def __init__(self, exp_dim, max_pos_exp, max_neg_exp, window_size, layer, device, profile_path, profile_dims, blocks=None, keys=None):
        super(VLPSilu, self).__init__(layer, device, profile_path, profile_dims, blocks, keys)
        self.exp_dim = exp_dim
        self.max_pos_exp = max_pos_exp
        self.max_neg_exp = max_neg_exp
        self.window_size = window_size
        self.build_lut()

    def reset_lut(self, exp_dim, max_pos_exp, max_neg_exp, window_size):
        self.exp_dim = exp_dim
        self.max_pos_exp = max_pos_exp
        self.max_neg_exp = max_neg_exp
        self.window_size = window_size
        self.build_lut()

    def build_lut(self):
        self.build_pos_lut()
        self.build_neg_lut()

    def build_pos_lut(self):
        # Mantissa dimension of virtual LUT
        mant_dim = 8

        # shift min_exp by max_exp
        self.pos_min_exp = self.max_pos_exp - (self.exp_dim - 1)

        # generate exponent values of LUT
        exp_values = torch.arange(self.exp_dim).reshape(self.exp_dim, 1)
        exp_table = exp_values.expand(self.exp_dim, mant_dim)
        exp_table = exp_table + self.pos_min_exp

        # generate mantissa values of LUT
        mant_values = torch.arange(mant_dim)
        mant_table = ((mant_values.expand(self.exp_dim, mant_dim) / 8) + 1)

        # combine exponent and mantissa values
        exp_table = exp_table.to(torch.int32)
        mant_table = mant_table.to(torch.float32)
        lookup_table = torch.ldexp(mant_table, exp_table)

        # apply exp to create LUT
        self.pos_lut = torch.nn.functional.silu(lookup_table).to(torch.bfloat16).to(self.device)

    def build_neg_lut(self):
        # Mantissa dimension of virtual LUT
        mant_dim = 8

        # shift min_exp by max_exp
        self.neg_min_exp = self.max_neg_exp - (self.exp_dim - 1)

        # generate exponent values of LUT
        exp_values = torch.arange(self.exp_dim).reshape(self.exp_dim, 1)
        exp_table = exp_values.expand(self.exp_dim, mant_dim)
        exp_table = exp_table + self.neg_min_exp

        # generate mantissa values of LUT
        mant_values = torch.arange(mant_dim)
        mant_table = ((mant_values.expand(self.exp_dim, mant_dim) / 8) + 1) * -1

        # combine exponent and mantissa values
        exp_table = exp_table.to(torch.int32)
        mant_table = mant_table.to(torch.float32)
        lookup_table = torch.ldexp(mant_table, exp_table)

        # apply exp to create LUT
        self.neg_lut = torch.nn.functional.silu(lookup_table).to(torch.bfloat16).to(self.device)

    def window_silu_approx(self, exp, mant):
        input_shape = exp.shape

        exp = exp.reshape(-1, input_shape[-1])
        mant = mant.reshape(-1, input_shape[-1])

        inter_shape = exp.shape

        if exp.shape[-1] % self.window_size != 0:
            padding = self.window_size - (exp.shape[-1] % self.window_size)

            padding_shape = [0] * len(exp.shape) * 2
            padding_shape[0] = padding

            pos_padded_exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=1000)
            neg_padded_exp = torch.nn.functional.pad(exp, pad=tuple(padding_shape), value=-1000)
            pos_padded_mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=1000)
            neg_padded_mant = torch.nn.functional.pad(mant, pad=tuple(padding_shape), value=-1000)
        else:
            pos_padded_exp = exp
            neg_padded_exp = exp
            pos_padded_mant = mant
            neg_padded_mant = mant

        del exp, mant

        pos_padded_exp = pos_padded_exp.reshape(pos_padded_exp.shape[0], pos_padded_exp.shape[1] // self.window_size, self.window_size)
        neg_padded_exp = neg_padded_exp.reshape(neg_padded_exp.shape[0], neg_padded_exp.shape[1] // self.window_size, self.window_size)
        pos_padded_mant = pos_padded_mant.reshape(pos_padded_mant.shape[0], pos_padded_mant.shape[1] // self.window_size, self.window_size)
        neg_padded_mant = neg_padded_mant.reshape(neg_padded_mant.shape[0], neg_padded_mant.shape[1] // self.window_size, self.window_size)

        # calculate pos min and max windows
        pos_max_exp_window = torch.max(pos_padded_exp, dim = -1, keepdim=True)[0].expand_as(pos_padded_exp)
        pos_max_exp_window = torch.where(pos_max_exp_window > self.max_pos_exp, self.max_pos_exp, pos_max_exp_window)
        pos_min_exp_window = pos_max_exp_window - 7
        pos_min_exp_window = torch.where(pos_min_exp_window < self.pos_min_exp, self.pos_min_exp, pos_min_exp_window)

        # calculate neg min and max windows
        neg_max_exp_window = torch.max(neg_padded_exp, dim = -1, keepdim=True)[0].expand_as(neg_padded_exp)
        neg_max_exp_window = torch.clamp(neg_max_exp_window, max=self.max_neg_exp)
        # cneg_max_exp_window = torch.where(neg_max_exp_window > self.max_neg_exp, self.max_neg_exp, neg_max_exp_window)
        neg_min_exp_window = neg_max_exp_window - 7
        neg_min_exp_window = torch.clamp(neg_min_exp_window, min=self.neg_min_exp)
        # neg_min_exp_window = torch.where(neg_min_exp_window < self.neg_min_exp, self.neg_min_exp, neg_min_exp_window)

        # compare to pos min and max values
        #pos_exp_window_max = torch.where(pos_padded_exp <= pos_max_exp_window, pos_padded_exp, pos_max_exp_window)
        pos_exp_window = torch.clamp(pos_padded_exp, max=pos_max_exp_window, min=pos_min_exp_window)
        #pos_exp_window = torch.where(pos_exp_window_max >= pos_min_exp_window, pos_exp_window_max, torch.where(pos_exp_window_max == -1000, pos_exp_window_max, pos_min_exp_window))
        
        pos_mant_window = torch.where(pos_padded_exp <= pos_max_exp_window, pos_padded_mant, 7)
        pos_mant_window = torch.where(pos_padded_exp >= pos_min_exp_window, pos_mant_window, 0)
        
        # compare to pos min and max values
        neg_exp_window = torch.clamp(neg_padded_exp, max=neg_max_exp_window, min=neg_min_exp_window)
        # neg_exp_window_max = torch.where(neg_padded_exp <= neg_max_exp_window, neg_padded_exp, neg_max_exp_window)
        # neg_exp_window = torch.where(neg_exp_window_max >= neg_min_exp_window, neg_exp_window_max, torch.where(neg_exp_window_max == -1000, neg_exp_window_max, neg_min_exp_window))
        
        neg_mant_window = torch.where(neg_padded_exp <= neg_max_exp_window, neg_padded_mant, 7)
        neg_mant_window = torch.where(neg_padded_exp >= neg_min_exp_window, neg_mant_window, 0)

        # adjust to index lut
        pos_exp_window = pos_exp_window - self.pos_min_exp
        neg_exp_window = neg_exp_window - self.neg_min_exp

        # Unpad and reshape tensors to original shape
        pos_exp_window = pos_exp_window.view(-1, input_shape[-1])
        pos_mant_window = pos_mant_window.view(-1, input_shape[-1])
        neg_exp_window = neg_exp_window.view(-1, input_shape[-1])
        neg_mant_window = neg_mant_window.view(-1, input_shape[-1])

        pos_exp_window = pos_exp_window[:inter_shape[0], :]
        pos_mant_window = pos_mant_window[:inter_shape[0], :]
        neg_exp_window = neg_exp_window[:inter_shape[0], :]
        neg_mant_window = neg_mant_window[:inter_shape[0], :]

        pos_exp_window = pos_exp_window.view(*input_shape)
        pos_mant_window = pos_mant_window.view(*input_shape)
        neg_exp_window = neg_exp_window.view(*input_shape)
        neg_mant_window = neg_mant_window.view(*input_shape)

        return pos_exp_window, pos_mant_window, neg_exp_window, neg_mant_window

    def lut_index(self, x, pos_exp, pos_mant, neg_exp, neg_mant, exp):

        silu = torch.zeros_like(x, dtype=torch.bfloat16, device=self.device)
        zero_tensor = torch.zeros_like(x, dtype=torch.bfloat16, device=self.device)

        # Masks
        positive = (x > 0)
        negative = (x < 0)
        pos_exp_greater = positive & (exp > self.max_pos_exp)
        pos_exp_less = positive & (exp < self.pos_min_exp)
        neg_exp_greater = negative & (exp > self.max_neg_exp)
        neg_exp_less = negative & (exp < self.neg_min_exp)
        zero_mask = (x == 0)

        # Apply LUT index masks
        silu[positive] = self.pos_lut[pos_exp[positive], pos_mant[positive]]
        silu[negative] = self.neg_lut[neg_exp[negative], neg_mant[negative]]

        # Apply conditions for out of bounds values
        silu[pos_exp_greater] = x[pos_exp_greater]
        silu[pos_exp_less] = zero_tensor[pos_exp_less]
        silu[neg_exp_greater] = zero_tensor[neg_exp_greater]
        silu[neg_exp_less] = zero_tensor[neg_exp_less]
        silu[zero_mask] = zero_tensor[zero_mask]

        return silu

    def nonlinear(self, x):
        # Split exponent and signed mantissa, bitshift mantissa to 4 bits (assumes leading 0).
        x = x.to(torch.bfloat16)
        mant, exp = torch.frexp(x)
        mant = torch.round(mant * 16)

        # Increment exponent where mantissa has overflow (i.e., mantissa is 16 / needs)
        exp = torch.where(torch.abs(mant) == 16, exp + 1, exp)
        exp = torch.where(x == 0, exp, exp - 1)

        # Convert mantissa to unsigned 3 bit integer
        mant = torch.abs(mant.to(torch.int32)) & 0x7

        # Increase exponent size for stability
        exp = exp.to(torch.int32)
        mant = mant.to(torch.int32)

        pos_exp, pos_mant, neg_exp, neg_mant = self.window_silu_approx(exp, mant)

        pos_mant = pos_mant.to(torch.int32)
        neg_mant = neg_mant.to(torch.int32)

        silu = self.lut_index(x, pos_exp, pos_mant, neg_exp, neg_mant, exp)

        return silu