def __getattr__(name):
    raise ImportError(f"librosa is a placeholder for Llama-only runs; librosa.{name} is unavailable")
