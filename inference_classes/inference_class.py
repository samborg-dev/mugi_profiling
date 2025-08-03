from datasets import load_dataset
from itertools import product
import torch
import gc
import os
import types
import pandas as pd
from transformers import activations

from custom_nonlinear.custom_approx import CustomSoftmax, CustomSilu, CustomGelu, CustomFastGelu
from custom_nonlinear.custom_eager import LlamaEager, VivitEager, WhisperEager
from custom_nonlinear.custom_forward import llama_forward, swin_forward, vivit_forward, whisper_forward
from custom_nonlinear.custom_nonlinear_functions.pwl_gelu_approx import PWLGelu
from custom_nonlinear.custom_nonlinear_functions.pwl_mobilenet_approx import PWLMobilenet
from custom_nonlinear.custom_nonlinear_functions.pwl_silu_approx import PWLSilu
from custom_nonlinear.custom_nonlinear_functions.pwl_softmax_approx import PWLSoftmax
from custom_nonlinear.custom_nonlinear_functions.taylor_softmax_approx import TaylorSoftmax
from custom_nonlinear.custom_nonlinear_functions.vlp_gelu_approx import VLPGelu
from custom_nonlinear.custom_nonlinear_functions.vlp_silu_approx import VLPSilu
from custom_nonlinear.custom_nonlinear_functions.vlp_softmax_approx import VLPSoftmax

class InferenceModel:
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        # Set device and multi-GPU support
        self.device = device
        self.use_multi_gpu = isinstance(device, dict) or (isinstance(device, str) and ',' in device)
        
        if self.use_multi_gpu:
            if isinstance(device, str):
                # Parse comma-separated device string: "cuda:0,cuda:1"
                self.device_list = [d.strip() for d in device.split(',')]
                self.primary_device = self.device_list[0]
            elif isinstance(device, dict):
                # Device mapping dictionary for specific layers
                self.device_map = device
                self.primary_device = list(device.values())[0]
            else:
                self.device_list = device
                self.primary_device = device[0]
        else:
            self.primary_device = device
            self.device_list = [device]
        
        # Initial dicts
        self.model_dict = model_dict
        self.nonlinear_dict = nonlinear_dict
        self.parameter_dict = parameter_dict

        # Dict configs
        self.dataset_parameters = model_dict.get('dataset')
        self.model_parameters = model_dict.get('model')
        self.inference_parameters = model_dict.get('parameters')
        self.nonlinear_parameters = model_dict.get('nonlinear')
        self.nonlinear_functions = nonlinear_dict.get('functions')
        self.nonlinear_function_parameters = nonlinear_dict.get('params')

        # Dict Items
        self.dataset_name = self.dataset_parameters.get('name')
        self.hf_path = self.dataset_parameters.get('hf_path')
        self.dataset_split = self.dataset_parameters.get('split')
        self.dataset_config = self.dataset_parameters.get('config')

        self.model_name = self.model_parameters.get('name')
        self.model_modality = self.model_parameters.get('modality')

        self.attn_op = self.nonlinear_parameters.get('attention')
        self.ffn_op = self.nonlinear_parameters.get('ffn')

        self.n_samples = parameter_dict.get('n_samples', 1)
        self.batch_size = self.inference_parameters.get('batch_size', 1)
        
        # Initialize DataFrame for collecting results
        self.df = None

    def load_streaming_dataset(self):
        if self.dataset_config:
            self.dataset = load_dataset(self.hf_path, self.dataset_config, split=self.dataset_split, streaming=True, trust_remote_code=True)
        else:
            self.dataset = load_dataset(self.hf_path, split=self.dataset_split, streaming=True, trust_remote_code=True)

    def process_batch(self, batch):
        return batch

    def batch_dataset(self):
        assert self.n_samples % self.batch_size == 0, 'Number of samples must be divisible by batch size.'

        batched_data = []
        for i in range(0, self.n_samples, self.batch_size):
            batch = self.inputs[i:i + self.batch_size]
            batch = self.process_batch(batch)
            batched_data.append(batch)
        self.inputs = batched_data

    def flatten_dict(self, d: dict) -> dict:
        flat_dict = {}
        for key, value in d.items():
            if isinstance(value, list) and len(value) == 1:
                flat_dict[key] = value[0]
            elif isinstance(value, (str, int, float, bool)) or value is None:
                flat_dict[key] = value
            else:
                raise ValueError(f'Value for key "{key}" is not a single-element list.')
        return flat_dict

    def dict_value_to_list(self, d: dict) -> dict:
        if d is None:
            return {}
        list_dict = {}
        for key, value in d.items():
            if not isinstance(value, list):
                list_dict[key] = [value]
            else:
                list_dict[key] = value
        return list_dict

    def parameter_combinations(self, d: dict) -> dict:
        (keys, values) = zip(*d.items())
        combination = list(product(*values))
        result = [dict(zip(keys, combo)) for combo in combination]
        return result

    def nonlinear_combinations(self, d: dict)-> dict:
        d = {k: v + [None] for k, v in d.items()}
        (keys, values) = zip(*d.items())
        combination = list(product(*values))
        result = [dict(zip(keys, combo)) for combo in combination]
        result = [r for r in result if not all(v is None for v in r.values())]
        return result

    def compute_loss(self, batch):
        return

    def run_inference(self):
        self.total_loss = 0.0
        self.num_batches = 0
        for batch in self.inputs:
            batched_loss = self.compute_loss(
                batch=batch
            )

            self.total_loss += batched_loss.item()
            self.num_batches += 1
            del batch, batched_loss
            torch.cuda.empty_cache()

        self.metric = self.compute_metric()

    def set_profiling_dims(self):
        self.profile_dims = -1

    def get_layer_device(self, layer_idx, total_layers=None):
        """Distribute layers across available GPUs"""
        if not self.use_multi_gpu:
            return self.device
        
        if hasattr(self, 'device_map'):
            # Use explicit device mapping if provided
            return self.device_map.get(f'layer_{layer_idx}', self.primary_device)
        
        # Distribute layers evenly across GPUs
        gpu_idx = layer_idx % len(self.device_list)
        return self.device_list[gpu_idx]

    def ensure_tensor_device(self, tensor, target_device):
        """Ensure tensor is on the correct device"""
        if tensor.device != torch.device(target_device):
            return tensor.to(target_device)
        return tensor

    def patch_model(self, function_name, attention_parameters={}, ffn_parameters={}, patch_attention=True, patch_ffn=True):
        torch.cuda.empty_cache()
        gc.collect()

        print(torch.cuda.memory_summary(device=self.primary_device, abbreviated=True))
        print(torch.cuda.get_device_properties(self.primary_device).total_memory)

        attention_default_classes = [CustomSoftmax]
        ffn_default_classes = [CustomSilu, CustomGelu, CustomFastGelu]

        attention_class = CustomSoftmax
        if self.ffn_op == 'silu':
            ffn_class = CustomSilu
        elif self.ffn_op == 'gelu':
            ffn_class = CustomGelu
        elif self.ffn_op == 'fast_gelu':
            ffn_class = CustomFastGelu

        if function_name == 'vlp':
            if patch_attention: attention_class = VLPSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = VLPSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = VLPGelu

        elif function_name == 'pwl':
            if patch_attention: attention_class = PWLSoftmax

            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLSilu
            elif (self.ffn_op == 'gelu' or self.ffn_op == 'fast_gelu') and patch_ffn: ffn_class = PWLGelu

        elif function_name == 'pwl_mobilenet':
            if self.ffn_op == 'silu' and patch_ffn: ffn_class = PWLMobilenet

        elif function_name == 'taylor':
            if patch_attention: attention_class = TaylorSoftmax

        if attention_class in attention_default_classes:
            attention_parameters = {}
        if ffn_class in ffn_default_classes:
            ffn_parameters = {}

        attn_path = f'{function_name}_{self.attn_op}' if patch_attention else f'torch_{self.attn_op}'
        ffn_path = f'{function_name}_{self.ffn_op}' if patch_ffn else f'torch_{self.ffn_op}'

        path = f'profile/{attn_path}_{ffn_path}/{self.model_name}/'

        os.makedirs(path, exist_ok=True)

        attention_parameters = attention_parameters if attention_parameters else {}
        ffn_parameters = ffn_parameters if ffn_parameters else {}

        if 'llama' in self.model_name:
            for i, layer in enumerate(self.model.model.layers):
                layer_device = self.get_layer_device(i, len(self.model.model.layers))
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims)
                eager_attn_fn = LlamaEager(nonlinear_object=attention_object)
                forward = llama_forward(eager_attn_fn)
                
                # Move layer to appropriate device
                layer.to(layer_device)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.mlp.act_fn = ffn_object
        
        elif 'whisper' in self.model_name:
            for i, layer in enumerate(self.model.model.encoder.layers):
                layer_device = self.get_layer_device(i)
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims)
                eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
                forward = whisper_forward(eager_attn_fn)
                
                layer.to(layer_device)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.activation_fn = ffn_object

            for i, layer in enumerate(self.model.model.decoder.layers):
                layer_device = self.get_layer_device(i + len(self.model.model.encoder.layers))
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims)
                eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
                forward = whisper_forward(eager_attn_fn)
                
                layer.to(layer_device)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.activation_fn = ffn_object
        elif 'swinv2' in self.model_name:
            layer_count = 0
            for i, block in enumerate(self.model.swinv2.encoder.layers):
                for j, layer in enumerate(block.blocks):
                    layer_device = self.get_layer_device(layer_count)
                    attention_object = attention_class(**attention_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims)
                    ffn_object = ffn_class(**ffn_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims)
                    forward = swin_forward(attention_object)
                    
                    layer.to(layer_device)
                    layer.attention.self.forward = types.MethodType(forward, layer.attention.self)
                    layer.intermediate.intermediate_act_fn = ffn_object
                    layer_count += 1
        
        elif 'vivit' in self.model_name:
            for i, layer in enumerate(self.model.vivit.encoder.layer):
                layer_device = self.get_layer_device(i)
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims)
                eager_attn_fn = VivitEager(nonlinear_object=attention_object)
                forward = vivit_forward(eager_attn_fn)

                layer.to(layer_device)
                layer.attention.attention.forward = types.MethodType(forward, layer.attention.attention)
                layer.intermediate.intermediate_act_fn = ffn_object

        
        torch.cuda.empty_cache()
        self.run_inference()
        torch.cuda.empty_cache()
        gc.collect()

        new_row = {
            'model': self.model_name,
            'modality': self.model_modality,
            'value': self.metric,
            'function_name': function_name,
            'patch_attention': patch_attention,
            'patch_ffn': patch_ffn,
            'attn_fn': attention_class.__name__,
            'ffn_fn': ffn_class.__name__
        }

        # Add attention parameters with prefixed column names to avoid conflicts
        if attention_parameters:
            for key, value in attention_parameters.items():
                new_row[f'attn_{key}'] = value
        
        # Add FFN parameters with prefixed column names to avoid conflicts
        if ffn_parameters:
            for key, value in ffn_parameters.items():
                new_row[f'ffn_{key}'] = value

        new_row = pd.DataFrame([new_row])

        if self.df is None:
            self.df = new_row
        else:
            self.df = pd.concat([self.df, new_row], axis=0, ignore_index=True)

    def loop_configuration(self):
        for function_name, function_operations in self.nonlinear_functions.items():
            if 'ffn' in function_operations:
                if self.ffn_op not in function_operations['ffn']:
                    function_operations.pop('ffn', None)
                else:
                    function_operations['ffn'] = [self.ffn_op]

            if not function_operations:
                continue

            nonlinear_combinations = self.nonlinear_combinations(function_operations) if function_name != 'torch' else [function_operations]

            for nonlinear_combination in nonlinear_combinations:
                nonlinear_combination = self.flatten_dict(nonlinear_combination)

                function_parameters = self.nonlinear_function_parameters.get(function_name)
                attention_parameters = function_parameters.get('attention') if function_parameters else None
                ffn_parameters = function_parameters.get('ffn') if function_parameters else None

                attn_op = nonlinear_combination.get('attention')
                ffn_op = nonlinear_combination.get('ffn')

                patch_attention = False
                patch_ffn = False

                attention_parameters = None if not attn_op else self.dict_value_to_list(attention_parameters) if attention_parameters else None
                ffn_parameters = None if not ffn_op else self.dict_value_to_list(ffn_parameters) if ffn_parameters else None

                attention_parameters = None if not attention_parameters else self.parameter_combinations(attention_parameters)
                ffn_parameters = None if not ffn_parameters else self.parameter_combinations(ffn_parameters)

                if not attn_op and not ffn_op:
                    continue
                elif (attn_op and not ffn_op) or (attn_op and ffn_op and not ffn_parameters):
                    patch_attention = True
                    if attention_parameters:
                        for attention_combination in attention_parameters:
                            self.patch_model(function_name, attention_parameters=attention_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                    else:
                        self.patch_model(function_name, patch_attention=patch_attention, patch_ffn=patch_ffn)

                elif (not attn_op and ffn_op) or (attn_op and ffn_op and not attention_parameters):
                    patch_ffn = True
                    if ffn_parameters:
                        for ffn_combination in ffn_parameters:
                            self.patch_model(function_name, ffn_parameters=ffn_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                    else:
                        self.patch_model(function_name, patch_attention=patch_attention, patch_ffn=patch_ffn)

                else:
                    patch_attention = True
                    patch_ffn = True
                    for attention_combination in attention_parameters:
                        for ffn_combination in ffn_parameters:
                            self.patch_model(function_name, attention_parameters=attention_combination, ffn_parameters=ffn_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                
                # Cleanup between nonlinear combinations
                torch.cuda.empty_cache()
                gc.collect()

        # Save the collected results to CSV
        if self.df is not None:
            csv_file = f'csv/{self.model_name}/metric.csv'
            os.makedirs(os.path.dirname(csv_file), exist_ok=True)
            self.df.to_csv(csv_file, index=False)

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        gc.collect()