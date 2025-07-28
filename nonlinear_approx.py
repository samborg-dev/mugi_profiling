from custom_approx import CustomSoftmax, CustomSilu, CustomGelu
from custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax
from custom_nonlinear_functions.vlp_silu_approx import VLPSilu
from custom_nonlinear_functions.vlp_gelu_approx import VLPGelu
from custom_nonlinear_functions.pwl_softmax_approx import PWLSoftmax
from custom_nonlinear_functions.pwl_silu_approx import PWLSilu
from custom_nonlinear_functions.pwl_gelu_approx import PWLGelu
from custom_nonlinear_functions.pwl_mobilenet_approx import PWLMobilenet
from custom_nonlinear_functions.taylor_softmax_approx import TaylorSoftmax

def default_softmax(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = CustomSoftmax(
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    return nonlinear_object

def default_silu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = CustomSilu(
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    return nonlinear_object

def default_gelu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = CustomGelu(
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    return nonlinear_object

def set_vlp_softmax(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = VLPSoftmax(
            exp_dim=config['exp_dim'],
            max_exp=config['max_exp'],
            min_exp=config['min_exp'],
            window_size=config['window_size'],
            lut_build=config['lut_build'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            exp_dim=config['exp_dim'],
            max_exp=config['max_exp'],
            min_exp=config['min_exp'],
            window_size=config['window_size'],
            lut_build=config['lut_build']
        )
    return nonlinear_object

def set_vlp_silu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = VLPSilu(
            exp_dim=config['exp_dim'],
            max_pos_exp=config['max_pos_exp'],
            max_neg_exp=config['max_neg_exp'],
            window_size=config['window_size'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            exp_dim=config['exp_dim'],
            max_pos_exp=config['max_pos_exp'],
            max_neg_exp=config['max_neg_exp'],
            window_size=config['window_size'],
        )
    return nonlinear_object

def set_vlp_gelu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = VLPGelu(
            exp_dim=config['exp_dim'],
            max_pos_exp=config['max_pos_exp'],
            max_neg_exp=config['max_neg_exp'],
            window_size=config['window_size'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            exp_dim=config['exp_dim'],
            max_pos_exp=config['max_pos_exp'],
            max_neg_exp=config['max_neg_exp'],
            window_size=config['window_size'],
        )
    return nonlinear_object

def set_pwl_softmax(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = PWLSoftmax(
            segments=config['segments'],
            segment_0=config['segment_0'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            segments=config['segments'],
            segment_0=config['segment_0']
        )
    return nonlinear_object

def set_pwl_silu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = PWLSilu(
            segments=config['segments'],
            segment_0=config['segment_0'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            segments=config['segments'],
            segment_0=config['segment_0']
        )
    return nonlinear_object

def set_pwl_gelu(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = PWLGelu(
            segments=config['segments'],
            segment_0=config['segment_0'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_lut(
            segments=config['segments'],
            segment_0=config['segment_0']
        )
    return nonlinear_object

def set_pwl_mobilenet(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = PWLMobilenet(
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile']
        )
    return nonlinear_object

def set_taylor_softmax(config, base_config, torch_nonlinear, nonlinear_object=None):
    if nonlinear_object is None:
        nonlinear_object = TaylorSoftmax(
            degree_center=config['degree_center'],
            degrees=config['degrees'],
            device=base_config['device'],
            path=config['path'],
            save_dims=base_config['save_dims'],
            profile=base_config['profile'],
            torch_nonlinear=torch_nonlinear
        )
    else:
        nonlinear_object.reset_taylor(
            degree_center=config['degree_center'],
            degrees=config['degrees']
        )
    return nonlinear_object