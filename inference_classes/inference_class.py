from datasets import load_dataset
from itertools import product
import torch
import gc
import os
import types
import pandas as pd
from tqdm import tqdm
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

from inference_classes.model_adapters import get_adapter

class InferenceModel:
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        # Set device
        self.device = device
        
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

    def _apply_adapter(self, attention_class, ffn_class, attention_parameters, ffn_parameters,
                       attention_keys, ffn_keys, path):
        """Model-agnostic instrumentation via the ModelAdapter registry. Returns #layers patched."""
        adapter = get_adapter(self.model)
        n = 0
        for site in adapter.layer_sites(self):
            if site.set_device:
                self.device = site.device

            attention_object = attention_class(**attention_parameters, layer=site.layer_idx, device=site.device, profile_path=path, profile_dims=site.profile_dims, keys=attention_keys, **site.keys)
            ffn_object = ffn_class(**ffn_parameters, layer=site.layer_idx, device=site.device, profile_path=path, profile_dims=site.profile_dims, keys=ffn_keys, **site.keys)

            forward = site.forward_builder(attention_object)
            site.attn_module.forward = types.MethodType(forward, site.attn_module)
            setattr(site.ffn_parent, site.ffn_attr, ffn_object)

            # the patch must actually take on the real module (guards typo'd paths / read-only attrs)
            assert site.attn_module.forward.__func__ is forward, f"attention patch did not take at layer {site.layer_idx}"
            assert getattr(site.ffn_parent, site.ffn_attr) is ffn_object, f"ffn patch did not take at layer {site.layer_idx}"
            n += 1
        return n

    def _apply_legacy(self, attention_class, ffn_class, attention_parameters, ffn_parameters,
                      attention_keys, ffn_keys, path):
        """Verbatim pre-refactor instrumentation, kept as oracle + revert switch. Returns #layers."""
        n = 0
        if 'llama' in self.model_name:
            for i, layer in enumerate(self.model.model.layers):
                layer_device = next(layer.parameters()).device
                if i == 0:
                    self.device = layer_device
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims, keys=attention_keys)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profiling_dims, keys=ffn_keys)
                eager_attn_fn = LlamaEager(nonlinear_object=attention_object)
                forward = llama_forward(eager_attn_fn)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.mlp.act_fn = ffn_object
                n += 1
        elif 'whisper' in self.model_name:
            for i, layer in enumerate(self.model.model.encoder.layers):
                layer_device = next(layer.parameters()).device
                if i == 0:
                    self.device = layer_device
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims, keys=attention_keys)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.source_profiling_dims, keys=ffn_keys)
                eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
                forward = whisper_forward(eager_attn_fn)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.activation_fn = ffn_object
                n += 1
            for i, layer in enumerate(self.model.model.decoder.layers):
                layer_device = next(layer.parameters()).device
                if i == 0:
                    self.device = layer_device
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims, keys=attention_keys)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.target_profiling_dims, keys=ffn_keys)
                eager_attn_fn = WhisperEager(nonlinear_object=attention_object)
                forward = whisper_forward(eager_attn_fn)
                layer.self_attn.forward = types.MethodType(forward, layer.self_attn)
                layer.activation_fn = ffn_object
                n += 1
        elif 'swinv2' in self.model_name:
            for i, block in enumerate(self.model.swinv2.encoder.layers):
                for j, layer in enumerate(block.blocks):
                    layer_device = next(layer.parameters()).device
                    if i == 0:
                        self.device = layer_device
                    attention_object = attention_class(**attention_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=attention_keys)
                    ffn_object = ffn_class(**ffn_parameters, layer=j, blocks=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=ffn_keys)
                    forward = swin_forward(attention_object)
                    layer.attention.self.forward = types.MethodType(forward, layer.attention.self)
                    layer.intermediate.intermediate_act_fn = ffn_object
                    n += 1
        elif 'vivit' in self.model_name:
            for i, layer in enumerate(self.model.vivit.encoder.layer):
                layer_device = next(layer.parameters()).device
                if i == 0:
                    self.device = layer_device
                attention_object = attention_class(**attention_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=attention_keys)
                ffn_object = ffn_class(**ffn_parameters, layer=i, device=layer_device, profile_path=path, profile_dims=self.profile_dims, keys=ffn_keys)
                eager_attn_fn = VivitEager(nonlinear_object=attention_object)
                forward = vivit_forward(eager_attn_fn)
                layer.attention.attention.forward = types.MethodType(forward, layer.attention.attention)
                layer.intermediate.intermediate_act_fn = ffn_object
                n += 1
        return n

    def patch_model(self, function_name, attention_parameters={}, ffn_parameters={}, patch_attention=True, patch_ffn=True):

        attention_keys = []
        if attention_parameters:
            for key, item in attention_parameters.items():
                attention_keys.append(key + '_' + str(item))

        ffn_keys = []
        if ffn_parameters:
            for key, item in ffn_parameters.items():
                ffn_keys.append(key + '_' + str(item))

        torch.cuda.empty_cache()
        gc.collect()

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

        path = f'profile/{self.model_name}/{attn_path}_{ffn_path}/'

        os.makedirs(path, exist_ok=True)

        attention_parameters = attention_parameters if attention_parameters else {}
        ffn_parameters = ffn_parameters if ffn_parameters else {}

        # Instrumentation. The adapter registry is the model-agnostic path; the verbatim legacy
        # if/elif (MUGI_USE_LEGACY_PATCH=1) is an instant revert switch + oracle. Whichever runs,
        # the patch is VERIFIED below: the profiler feeds the LUT window config, which sets the
        # paper's accuracy/efficiency numbers, so a wrong/zero/partial patch must fail loudly,
        # never silently drift.
        if os.environ.get("MUGI_USE_LEGACY_PATCH"):
            n_patched = self._apply_legacy(attention_class, ffn_class, attention_parameters,
                                           ffn_parameters, attention_keys, ffn_keys, path)
        else:
            n_patched = self._apply_adapter(attention_class, ffn_class, attention_parameters,
                                            ffn_parameters, attention_keys, ffn_keys, path)

        expected = get_adapter(self.model).expected_count(self.model)
        if n_patched == 0:
            raise RuntimeError(f"patch_model instrumented 0 layers for model {self.model_name!r}.")
        if n_patched != expected:
            raise RuntimeError(
                f"patch_model instrumented {n_patched} layers but the config independently "
                f"declares {expected} for model {self.model_name!r} - instrumentation is "
                f"incomplete/incorrect; aborting to avoid a silently-wrong window config."
            )

        
        torch.cuda.empty_cache()
        # print('pre_inference')
        # for i in range(torch.cuda.device_count()):
        #     device = torch.device(f"cuda:{i}")
        #     print(f"\n=== CUDA Device {i}: {torch.cuda.get_device_name(device)} ===")
        #     print(torch.cuda.memory_summary(device=device, abbreviated=True))
        # print()
        self.run_inference()
        # print('post_inference')
        # for i in range(torch.cuda.device_count()):
        #     device = torch.device(f"cuda:{i}")
        #     print(f"\n=== CUDA Device {i}: {torch.cuda.get_device_name(device)} ===")
        #     print(torch.cuda.memory_summary(device=device, abbreviated=True))
        # print()
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
        for function_name, function_operations in tqdm(self.nonlinear_functions.items(), desc='Patching configurations'):
            if 'ffn' in function_operations:
                if self.ffn_op not in function_operations['ffn']:
                    function_operations.pop('ffn', None)
                else:
                    function_operations['ffn'] = [self.ffn_op]

            if not function_operations:
                continue

            nonlinear_combinations = self.nonlinear_combinations(function_operations) if function_name != 'torch' else [function_operations]

            for nonlinear_combination in tqdm(nonlinear_combinations, desc=f'Processing {function_name} combinations'):
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
                    pass
                    # patch_attention = True
                    # patch_ffn = True
                    # for attention_combination in tqdm(attention_parameters, desc='Attention combinations'):
                    #     for ffn_combination in tqdm(ffn_parameters, desc='FFN combinations'):
                    #         self.patch_model(function_name, attention_parameters=attention_combination, ffn_parameters=ffn_combination, patch_attention=patch_attention, patch_ffn=patch_ffn)
                
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