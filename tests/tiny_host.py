import math

import pytest
import torch

transformers = pytest.importorskip("transformers")

from inference_classes.inference_class import InferenceModel
from profiling_api.evaluate import WindowEvalHarness

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"


class TinyHost(InferenceModel):
    def __init__(self, model, tmp_path, n_batches=2, batch_size=1, seq_len=16):
        self.device = 'cpu'
        self.model = model
        self.model_name = 'llama'
        self.model_modality = 'nlp'
        self.attn_op = 'softmax'
        self.ffn_op = 'silu'
        self.df = None
        self.nonlinear_sites = []
        self.max_length = seq_len
        self.profiling_dims = [seq_len - 1]
        self._profile_root = str(tmp_path)

        g = torch.Generator().manual_seed(0)
        vocab = model.config.vocab_size
        self.inputs = []
        for _ in range(n_batches):
            batch = []
            for _ in range(batch_size):
                ids = torch.randint(0, vocab, (seq_len,), generator=g)
                batch.append({'input_ids': ids,
                              'attention_mask': torch.ones_like(ids)})
            self.inputs.append(batch)

    def compute_metric(self):
        return math.exp(self.total_loss / self.num_batches)

    def compute_loss(self, batch):
        input_ids = torch.stack([ex['input_ids'] for ex in batch])
        attention_mask = torch.stack([ex['attention_mask'] for ex in batch]).bool()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             labels=input_ids, use_cache=False)
        return out.loss


def load_tiny(attn_implementation='eager'):
    try:
        return transformers.AutoModelForCausalLM.from_pretrained(
            TINY_LLAMA, attn_implementation=attn_implementation)
    except Exception as e:
        pytest.skip(f"could not load {TINY_LLAMA}: {type(e).__name__}")


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = load_tiny()
    model.eval()
    return TinyHost(model, tmp_path)


@pytest.fixture
def harness(host):
    h = WindowEvalHarness(host)
    h.setup()
    return h
