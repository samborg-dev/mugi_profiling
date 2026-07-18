import torch
import os
import yaml


def get_subdirs(path):
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return dirs

def get_seq_len(path):
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_values = [int((f.split('.')[0]).split('_')[-1]) for f in files if f.endswith('.pt')]
    file_value = max(file_values) if file_values else 0
    return f'seq_len_{file_value}.pt'

def has_subdirs(path):
    return any(os.path.isdir(os.path.join(path, d)) for d in os.listdir(path))

def topk_window_until_threshold(counts: torch.Tensor, threshold: float = 0.99):
    if counts.dim() != 1:
        raise ValueError("counts must be a 1D tensor")

    total = counts.sum().item()
    if total == 0:
        return torch.tensor([], dtype=torch.long), 0.0

    # Sort by value (descending)
    values, indices = torch.sort(counts, descending=True)

    # Cumulative sum
    cumsum = torch.cumsum(values, dim=0)
    frac = cumsum / total

    # Find smallest window where threshold is reached
    window_size = torch.nonzero(frac >= threshold, as_tuple=True)[0][0].item() + 1

    selected_indices = indices[:window_size]
    return selected_indices, frac[window_size - 1].item()

def profile_tensor(path, window_size=8):

    layer = (path.split('/')[-2]).split('_')[-1]
    nonlinear = (path.split('/')[-5])

    tensor = torch.load(path, map_location='cpu')
    tensor = tensor[1:31]

    tensor = (tensor / tensor.sum()) * 100

    if nonlinear == 'softmax':
        threshold = 0.92
    else:
        threshold = 0.5
    topk_indices, topk_tensor = topk_window_until_threshold(tensor, threshold=threshold)
    
    while topk_indices.shape[0] > 5:
        threshold -= 0.01
        topk_indices, topk_tensor = topk_window_until_threshold(tensor, threshold=threshold)

    while topk_indices.shape[0] < 4:
        threshold += 0.01
        topk_indices, topk_tensor = topk_window_until_threshold(tensor, threshold=threshold)

    topk_max = topk_indices.max().item()
    topk_min = topk_indices.min().item()

    topk_tensor = tensor[topk_min:topk_max + 1]

    
    topk_max -= 15
    topk_min -= 15

    argmax_value = tensor.argmax().item() - 15

    window_sums = torch.tensor([tensor[i:i + window_size].sum() for i in range(len(tensor) - window_size + 1)])
    max_sum = window_sums.argmax().item()

    tensor_windowed = tensor[max_sum:max_sum + window_size]

    max_idx = tensor_windowed.argmax()
    max_value_idx = (max_sum + (window_size - 1)) - 15
    min_value_idx = (max_sum - 15)

    indices = torch.arange(len(tensor_windowed))
    centroid = (tensor_windowed * indices).sum() / tensor_windowed.sum()

    max_value = max_value_idx
    min_value = min_value_idx

    max_idx = max_value + 15
    min_idx = min_value + 15

    max_sum = torch.sum(tensor[max_idx - window_size:max_idx])
    min_sum = torch.sum(tensor[min_idx:min_idx + window_size])

    # if nonlinear == 'softmax':
    #     cluster = 'min_cluster'
    # else:
    #     cluster = 'max_cluster'

    # cluster = 'min_cluster'

    # if max_sum < min_sum:
    #     cluster = 'max_cluster'
    # elif min_sum < max_sum:
    #     cluster = 'min_cluster'

    layer = int(layer)
    if layer == 0:
        if nonlinear == 'softmax':
            topk_max = 10
            topk_min = 10
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 0
            topk_min = 0
            cluster = 'max_cluster'
    elif layer > 0 and layer <= 1:
        if nonlinear == 'softmax':
            topk_max = 4
            topk_min = 4
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 0
            topk_min = 0
            cluster = 'max_cluster'
    elif layer > 1 and layer <= 2:
        if nonlinear == 'softmax':
            topk_max = 1
            topk_min = 1
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 1
            topk_min = 1
            cluster = 'max_cluster'
    elif layer > 2 and layer <=5:
        if nonlinear == 'softmax':
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 5 and layer <=8:
        if nonlinear == 'softmax':
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 8 and layer <=15:
        if nonlinear == 'softmax':
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 15 and layer <= 19:
        if nonlinear == 'softmax':
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 19 and layer <= 23:
        if nonlinear == 'softmax':
            topk_max = 4
            topk_min = 2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 23 and layer <= 25:
        if nonlinear == 'softmax':
            topk_max = 4
            topk_min = -2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 25 and layer <= 28:
        if nonlinear == 'softmax':
            topk_max = 4
            topk_min = -2
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer > 28 and layer < 30:
        if nonlinear == 'softmax':
            topk_max = 3
            topk_min = -6
            cluster = 'max_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer == 30:
        if nonlinear == 'softmax':
            topk_max = 3
            topk_min = -5
            cluster = 'min_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = 2
            cluster = 'max_cluster'
    elif layer == 31:
        if nonlinear == 'softmax':
            topk_max = 3
            topk_min = -11
            cluster = 'min_cluster'
        elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
            topk_max = 2
            topk_min = -6
            cluster = 'max_cluster'
    # else:
    #     if nonlinear == 'softmax':
    #         topk_max = -5
    #         topk_min = -5
    #         cluster = 'min_cluster'
    #     elif nonlinear in ['gelu', 'silu', 'fast_gelu']:
    #         topk_max = 2
    #         topk_min = 2
    #         cluster = 'max_cluster'

    #topk_tensor = topk_tensor.to(torch.float32)

    mean = topk_tensor.mean().item()
    median = topk_tensor.median().item()

    # if mean > median:
    #     cluster = 'min_cluster'
    # else:
    #     cluster = 'max_cluster'

    # if nonlinear == 'softmax':
    #     cluster = 'min_cluster'
    # else:
    #     cluster = 'max_cluster'

    # if nonlinear == 'softmax':
    #     cluster = 'min_cluster'

    data_dict = {
        'argmax_value': argmax_value,
        'tensor': tensor_windowed.tolist(),
        'max_exp': topk_max,
        'min_exp': topk_min,
        'max_exp_unclamped': max_value_idx,
        'min_exp_unclamped': min_value_idx,
        'mean': mean,
        'median': median,
        'centroid': centroid.item(),
        'cluster': cluster
    }

    return data_dict

def profile_list(path_list, model_name, nonlinear_op):
    for path in path_list:
        seq_len = get_seq_len(path)
        new_path = path.split('exp_dist/')[-1]
        save_path = os.path.join('distribution', str(model_name), str(nonlinear_op), new_path)
        tensor_path = os.path.join(path, seq_len)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_dict = profile_tensor(tensor_path)
        save_file = os.path.join(save_path, 'profile.yaml')
        with open(save_file, 'w') as f:
            yaml.dump(data_dict, f)

def loop_through_subdirs(path):

    paths = []

    if has_subdirs(path):
        subdirs = get_subdirs(path)
        for subdir in subdirs:
            subdir_path = os.path.join(path, subdir)
            new_paths = loop_through_subdirs(subdir_path)
            if isinstance(new_paths, list):
                paths.extend(new_paths)
            else:
                paths.append(new_paths)
        return paths
    else:
        return path
    
def analyze_profile():
    base_path = 'profile'
    #company_list = ['google', 'meta-llama', 'openai', 'microsoft']
    company_list = ['meta-llama']

    google_list = ['vivit-b-16x2']
    #meta_llama_list = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-3.1-405B']
    meta_llama_list = ['Llama-2-7b-hf']
    openai_list = ['whisper-tiny', 'whisper-base', 'whisper-small', 'whisper-medium', 'whisper-large']
    #openai_list = ['whisper-tiny']
    microsoft_list = ['swinv2-tiny-patch4-window8-256', 'swinv2-base-patch4-window8-256', 'swinv2-small-patch4-window8-256', 'swinv2-large-patch4-window12to16-192to256-22kto1k-ft']
    pre_post = 'pre_'
    dist = 'exp_dist'

    for company in company_list:
        if company == 'google':
            model_list = google_list
            activation = 'fast_gelu'
            nonlinear_list = ['softmax', 'gelu']
        elif company == 'meta-llama':
            model_list = meta_llama_list
            activation = 'silu'
            nonlinear_list = ['silu', 'softmax']
        elif company == 'openai':
            model_list = openai_list
            activation = 'gelu'
            nonlinear_list = ['softmax', 'gelu']
        elif company == 'microsoft':
            model_list = microsoft_list
            activation = 'gelu'
            nonlinear_list = ['softmax', 'gelu']

        for model in model_list:
            for nonlinear in nonlinear_list:
                path = os.path.join(base_path, company, model, f'torch_softmax_torch_{activation}', nonlinear, f'{pre_post}{nonlinear}', dist)
                
                tensor_paths = loop_through_subdirs(path)
                profile_list(tensor_paths, model, nonlinear)
                
def create_nonlinear_config(path, model_name):
    subdirs = get_subdirs(path)
    nonlinear_dict = {
        'layer_config': True,
        'functions': {
            'vlp': {
                'attention': ['softmax'],
                'ffn': ['gelu', 'silu', 'fast_gelu']
            }
        },
        'params': {}
    }
    
    for subdir in subdirs:
        subsubdirs = get_subdirs(os.path.join(path, subdir))
        for subsubdir in subsubdirs:
            if 'swin' not in model_name:
                if subdir == 'softmax':
                    profile_dict = yaml.safe_load(open(os.path.join(path, subdir, subsubdir, 'profile.yaml')))
                    layer = subsubdir.split('_')[-1]

                    lut_build = 'max' if profile_dict['cluster'] == 'max_cluster' else 'min'

                    if layer not in nonlinear_dict['params']:
                        nonlinear_dict['params'][layer] = {
                            'vlp': {
                                'attention': {
                                    'exp_dim': 16,
                                    'max_exp': profile_dict['max_exp'],
                                    'min_exp': profile_dict['min_exp'],
                                    'window_size': 32,
                                    'lut_build': lut_build
                                }
                            }
                        }
                    else:
                        nonlinear_dict['params'][layer]['vlp']['attention'] = {
                            'exp_dim': 16,
                            'max_exp': profile_dict['max_exp'],
                            'min_exp': profile_dict['min_exp'],
                            'window_size': 32,
                            'lut_build': lut_build
                        }

                elif subdir in ['gelu', 'silu', 'fast_gelu']:
                    profile_dict = yaml.safe_load(open(os.path.join(path, subdir, subsubdir, 'profile.yaml')))
                    layer = subsubdir.split('_')[-1]

                    lut_build = 'max' if profile_dict['cluster'] == 'max_cluster' else 'min'

                    if layer not in nonlinear_dict['params']:
                        nonlinear_dict['params'][layer] = {
                            'vlp': {
                                'ffn': {
                                    'exp_dim': 16,
                                    'max_pos_exp': profile_dict['max_exp'],
                                    'min_pos_exp': profile_dict['min_exp'],
                                    'window_size': 32,
                                    'lut_build': lut_build
                                }
                            }
                        }
                    else:
                        nonlinear_dict['params'][layer]['vlp']['ffn'] = {
                            'exp_dim': 16,
                            'max_pos_exp': profile_dict['max_exp'],
                            'min_pos_exp': profile_dict['min_exp'],
                            'window_size': 32,
                            'lut_build': lut_build
                        }
            else:
                layer = subsubdir.split('_')[-1]
                for subsubsubdir in get_subdirs(os.path.join(path, subdir, subsubdir)):
                    if subdir == 'softmax':
                        profile_dict = yaml.safe_load(open(os.path.join(path, subdir, subsubdir, subsubsubdir, 'profile.yaml')))
                        block = subsubsubdir.split('_')[-1]

                        lut_build = 'max' if profile_dict['cluster'] == 'max_cluster' else 'min'

                        if layer not in nonlinear_dict['params']:
                            nonlinear_dict['params'][layer] = {
                                block: {
                                    'vlp': {
                                        'attention': {
                                            'exp_dim': 16,
                                            'max_exp': profile_dict['max_exp'],
                                            'min_exp': profile_dict['min_exp'],
                                            'window_size': 32,
                                            'lut_build': lut_build
                                        }
                                    }
                                }
                            }
                        else:
                            if block not in nonlinear_dict['params'][layer]:
                                nonlinear_dict['params'][layer][block] = {
                                    'vlp': {
                                        'attention': {
                                            'exp_dim': 16,
                                            'max_exp': profile_dict['max_exp'],
                                            'min_exp': profile_dict['min_exp'],
                                            'window_size': 32,
                                            'lut_build': lut_build
                                        }
                                    }
                                }
                            else:
                                nonlinear_dict['params'][layer][block]['vlp']['attention'] = {
                                    'exp_dim': 16,
                                    'max_exp': profile_dict['max_exp'],
                                    'min_exp': profile_dict['min_exp'],
                                    'window_size': 32,
                                    'lut_build': lut_build
                                }
                                

                    elif subdir in ['gelu', 'silu', 'fast_gelu']:
                        profile_dict = yaml.safe_load(open(os.path.join(path, subdir, subsubdir, subsubsubdir, 'profile.yaml')))
                        block = subsubsubdir.split('_')[-1]

                        lut_build = 'max' if profile_dict['cluster'] == 'max_cluster' else 'min'

                        if layer not in nonlinear_dict['params']:
                            nonlinear_dict['params'][layer] = {
                                block: {
                                    'vlp': {
                                        'ffn': {
                                            'exp_dim': 16,
                                            'max_pos_exp': profile_dict['max_exp'],
                                            'min_pos_exp': profile_dict['min_exp'],
                                            'window_size': 32,
                                            'lut_build': lut_build
                                        }
                                    }
                                }
                            }
                        else:
                            if block not in nonlinear_dict['params'][layer]:
                                nonlinear_dict['params'][layer][block] = {
                                    'vlp': {
                                        'ffn': {
                                            'exp_dim': 16,
                                            'max_pos_exp': profile_dict['max_exp'],
                                            'min_pos_exp': profile_dict['min_exp'],
                                            'window_size': 32,
                                            'lut_build': lut_build
                                        }
                                    }
                                }
                            else:
                                nonlinear_dict['params'][layer][block]['vlp']['ffn'] = {
                                    'exp_dim': 16,
                                    'max_pos_exp': profile_dict['max_exp'],
                                    'min_pos_exp': profile_dict['min_exp'],
                                    'window_size': 32,
                                    'lut_build': lut_build
                                }
    
    return nonlinear_dict

def analyze_analysis():
    base_path = 'distribution'
    #company_list = ['google', 'meta-llama', 'openai', 'microsoft']
    company_list = ['meta-llama']

    google_list = ['vivit-b-16x2']
    #meta_llama_list = ['Llama-2-7b-hf', 'Llama-2-13b-hf', 'Llama-2-70b-hf', 'Llama-3.1-8B', 'Llama-3.1-70B', 'Llama-3.1-405B']
    meta_llama_list = ['Llama-2-7b-hf']
    openai_list = ['whisper-tiny', 'whisper-base', 'whisper-small', 'whisper-medium', 'whisper-large']
    #openai_list = ['whisper-tiny']
    microsoft_list = ['swinv2-tiny-patch4-window8-256', 'swinv2-base-patch4-window8-256', 'swinv2-small-patch4-window8-256', 'swinv2-large-patch4-window12to16-192to256-22kto1k-ft']

    for company in company_list:
        if company == 'google':
            model_list = google_list
        elif company == 'meta-llama':
            model_list = meta_llama_list
        elif company == 'openai':
            model_list = openai_list
        elif company == 'microsoft':
            model_list = microsoft_list

        for model in model_list:
            path = os.path.join(base_path, model)
            nonlinear_config = create_nonlinear_config(path, model)
            save_path = os.path.join('nonlinear_configs', model)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_file = os.path.join(save_path, 'nonlinear_config.yaml')
            with open(save_file, 'w') as f:
                yaml.dump(nonlinear_config, f)
                        

if __name__ == '__main__':
    analyze_profile()
    analyze_analysis()