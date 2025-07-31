import torch
import gc
import shutil
from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, AutoModelForImageClassification

from inference_classes.inference_class import InferenceModel

class VisionModel(InferenceModel):
    def __init__(self, model_dict, nonlinear_dict, parameter_dict, device):
        super().__init__(model_dict, nonlinear_dict, parameter_dict, device)

    def load_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name, torch_dtype=torch.float16, attn_implementation='eager').to(self.device)
        #self.max_length = self.model.config.max_source_positions

    def process_dataset(self):
        self.subset = list(self.dataset.take(self.n_samples))
        self.inputs = []

        for example in self.subset:
            img = example["image"] if 'image' in example else example['jpg']

            inputs = self.processor(
                images=img,
                return_tensors="pt"
            )

            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
            label = example["label"] if 'label' in example else example['cls']

            processed_example = {
                "pixel_values": inputs["pixel_values"],
                "labels": torch.tensor([label], dtype=torch.long),
                "label": label
            }

            self.inputs.append(processed_example)

    def compute_metric(self):
        return self.total_loss / self.num_batches
    
    def compute_loss(self, batch):
        pixel_values = torch.stack([ex["pixel_values"].squeeze(0) for ex in batch]).to(self.model.device).to(torch.float16)
        labels = torch.stack([ex["labels"] for ex in batch]).squeeze(-1).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, labels=labels)

        del pixel_values, labels
        loss = outputs.loss
        return loss
    
    def cleanup(self):
        del self.model
        del self.processor
        del self.inputs
        del self.dataset
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        model_cache_path = snapshot_download(self.model_name, local_files_only=True)
        shutil.rmtree(model_cache_path, ignore_errors=True)

        gc.collect()