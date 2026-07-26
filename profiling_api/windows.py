from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Iterator, Mapping, Tuple

import yaml

ATTENTION = "attention"
FFN = "ffn"
SITES = (ATTENTION, FFN)

DEFAULT_GROUP_SIZE = 32


@dataclass(frozen=True)
class SoftmaxWindow:
    exp_dim: int
    anchor: int
    anchor_side: str = "max"
    group_size: int = DEFAULT_GROUP_SIZE

    def __post_init__(self):
        if self.exp_dim < 1:
            raise ValueError(f"exp_dim must be >= 1, got {self.exp_dim}")
        if self.anchor_side not in ("max", "min"):
            raise ValueError(f"anchor_side must be 'max' or 'min', got {self.anchor_side!r}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def max_exp(self) -> int:
        return self.anchor if self.anchor_side == "max" else self.anchor + (self.exp_dim - 1)

    @property
    def min_exp(self) -> int:
        return self.anchor - (self.exp_dim - 1) if self.anchor_side == "max" else self.anchor

    @property
    def covers(self) -> range:
        return range(self.min_exp, self.max_exp + 1)

    def to_kwargs(self) -> dict:
        return dict(
            exp_dim=self.exp_dim,
            max_exp=self.max_exp,
            min_exp=self.min_exp,
            window_size=self.group_size,
            lut_build=self.anchor_side,
        )

    @classmethod
    def from_profile(cls, max_exp: int, min_exp: int, cluster: str,
                     exp_dim: int = 16, group_size: int = DEFAULT_GROUP_SIZE) -> "SoftmaxWindow":
        side = "max" if cluster == "max_cluster" else "min"
        return cls(exp_dim=exp_dim,
                   anchor=max_exp if side == "max" else min_exp,
                   anchor_side=side,
                   group_size=group_size)


@dataclass(frozen=True)
class FfnWindow:
    exp_dim: int
    pos_anchor: int
    neg_anchor: int
    group_size: int = DEFAULT_GROUP_SIZE

    def __post_init__(self):
        if self.exp_dim < 1:
            raise ValueError(f"exp_dim must be >= 1, got {self.exp_dim}")
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def pos_min_exp(self) -> int:
        return self.pos_anchor - (self.exp_dim - 1)

    @property
    def neg_min_exp(self) -> int:
        return self.neg_anchor - (self.exp_dim - 1)

    def to_kwargs(self) -> dict:
        return dict(
            exp_dim=self.exp_dim,
            max_pos_exp=self.pos_anchor,
            max_neg_exp=self.neg_anchor,
            window_size=self.group_size,
        )

    @classmethod
    def from_profile(cls, max_exp: int, exp_dim: int = 16,
                     group_size: int = DEFAULT_GROUP_SIZE) -> "FfnWindow":
        return cls(exp_dim=exp_dim, pos_anchor=max_exp, neg_anchor=max_exp,
                   group_size=group_size)


def window_for_site(site: str, **fields):
    if site == ATTENTION:
        return SoftmaxWindow(**fields)
    if site == FFN:
        return FfnWindow(**fields)
    raise ValueError(f"unknown site {site!r}; expected one of {SITES}")


class WindowAssignment:
    def __init__(self, windows: Mapping[Tuple[int, str], object] = None):
        self._w: Dict[Tuple[int, str], object] = {}
        for key, window in (windows or {}).items():
            self.set(*key, window)

    @classmethod
    def uniform(cls, n_layers: int, attention=None, ffn=None) -> "WindowAssignment":
        a = cls()
        for layer in range(n_layers):
            if attention is not None:
                a.set(layer, ATTENTION, attention)
            if ffn is not None:
                a.set(layer, FFN, ffn)
        return a

    def with_override(self, layer: int, site: str, window) -> "WindowAssignment":
        out = WindowAssignment(self._w)
        out.set(layer, site, window)
        return out

    def with_field(self, layer: int, site: str, **changes) -> "WindowAssignment":
        return self.with_override(layer, site, replace(self.get(layer, site), **changes))

    def set(self, layer: int, site: str, window) -> None:
        if site not in SITES:
            raise ValueError(f"unknown site {site!r}; expected one of {SITES}")
        expected = SoftmaxWindow if site == ATTENTION else FfnWindow
        if not isinstance(window, expected):
            raise TypeError(f"site {site!r} needs a {expected.__name__}, "
                            f"got {type(window).__name__}")
        self._w[(int(layer), site)] = window

    def get(self, layer: int, site: str):
        return self._w[(int(layer), site)]

    def get_or_none(self, layer: int, site: str):
        return self._w.get((int(layer), site))

    @property
    def layers(self):
        return sorted({layer for layer, _ in self._w})

    def items(self) -> Iterator[Tuple[int, str, object]]:
        for (layer, site) in sorted(self._w, key=lambda k: (k[0], k[1])):
            yield layer, site, self._w[(layer, site)]

    def __len__(self):
        return len(self._w)

    def __eq__(self, other):
        return isinstance(other, WindowAssignment) and self._w == other._w

    def __repr__(self):
        return f"WindowAssignment({len(self._w)} windows over {len(self.layers)} layers)"

    def to_dict(self) -> dict:
        out: Dict[str, dict] = {}
        for layer, site, window in self.items():
            out.setdefault(str(layer), {})[site] = asdict(window)
        return out

    @classmethod
    def from_dict(cls, data: Mapping) -> "WindowAssignment":
        a = cls()
        for layer, sites in data.items():
            for site, fields in sites.items():
                a.set(int(layer), site, window_for_site(site, **fields))
        return a

    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_yaml(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump({"window_assignment": self.to_dict(),
                            "digest": self.digest()}, f, sort_keys=True)
        return path

    @classmethod
    def from_yaml(cls, path: str) -> "WindowAssignment":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data["window_assignment"])
