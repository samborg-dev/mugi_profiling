import gc
import os

import torch

from utils import validate_config
from inference_classes.audio_inference import AudioModel
from inference_classes.npl_inference import NLPModel
from inference_classes.video_inference import VideoModel
from inference_classes.vision_inference import VisionModel

from profiling_api.config import ProfileConfig


_MODALITY_CLASSES = {
    "nlp": NLPModel,
    "audio": AudioModel,
    "vision": VisionModel,
    "video": VideoModel,
}


class ModelLoader:
    def load(self, cfg: ProfileConfig):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modality = validate_config(cfg.model_dict, cfg.nonlinear_dict, cfg.parameter_dict)
        model_cls = _MODALITY_CLASSES.get(modality)
        if model_cls is None:
            raise ValueError(f"Unsupported modality: {modality!r}")

        model = model_cls(cfg.model_dict, cfg.nonlinear_dict, cfg.parameter_dict, device)

        model.csv_file = f"csv/{model.model_name}/metric.csv"
        if os.path.exists(model.csv_file):
            os.remove(model.csv_file)
        else:
            os.makedirs(os.path.dirname(model.csv_file), exist_ok=True)
        model.df = None

        model.load_model()
        model.load_streaming_dataset()
        model.process_dataset()
        model.batch_dataset()
        model.set_profiling_dims()
        return model


class ProfilingRunner:
    def run(self, model) -> None:
        model.loop_configuration()


class DistributionStore:
    def __init__(self, cfg: ProfileConfig):
        self.model_name = cfg.model_name
        self.root = os.path.join("profile", cfg.model_name)

    def exists(self) -> bool:
        return os.path.isdir(self.root)

    def tensor_dirs(self):
        from profile_distribution import loop_through_subdirs
        if not self.exists():
            return []
        found = loop_through_subdirs(self.root)
        return found if isinstance(found, list) else [found]


class ProfilingPipeline:
    def __init__(self, cfg: ProfileConfig):
        self.cfg = cfg
        self.loader = ModelLoader()
        self.runner = ProfilingRunner()

    def run(self) -> dict:
        cfg = self.cfg
        from profiling_api.emit import ConfigEmitter, ArchxWorkloadEmitter

        model = self.loader.load(cfg)
        try:
            self.runner.run(model)
            store = DistributionStore(cfg)

            nonlinear_config_path = ConfigEmitter(cfg).emit(store)
            archx_workload_path = ArchxWorkloadEmitter(cfg).emit(model)

            return {
                "profile_dir": store.root,
                "nonlinear_config": nonlinear_config_path,
                "archx_workload": archx_workload_path,
            }
        finally:
            model.cleanup()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
