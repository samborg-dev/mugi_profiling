from profiling_api.config import ProfileConfig, DatasetSpec

__all__ = ["ProfileConfig", "DatasetSpec", "ProfilingPipeline"]


def __getattr__(name):
    if name == "ProfilingPipeline":
        from profiling_api.pipeline import ProfilingPipeline
        return ProfilingPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
