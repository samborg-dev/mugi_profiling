from __future__ import annotations

from typing import Iterable

from profiling_api.windows import ATTENTION, FFN, SITES, WindowAssignment


def set_profiling(host, enabled: bool) -> int:
    n = 0
    for entry in host.nonlinear_sites:
        for site in (ATTENTION, FFN):
            obj = entry.get(site)
            if obj is not None and hasattr(obj, 'profiling_enabled'):
                obj.profiling_enabled = enabled
                n += 1
    return n


def _reset(obj, window, layer: int, site: str) -> None:
    reset_lut = getattr(obj, 'reset_lut', None)
    if reset_lut is None:
        raise TypeError(
            f"layer {layer} {site} is a {type(obj).__name__}, which has no reset_lut; "
            f"the model must be patched with a VLP approximation before windows can "
            f"be re-parameterised."
        )
    reset_lut(**window.to_kwargs())


def apply_assignment(host, assignment: WindowAssignment, sites: Iterable[str] = SITES) -> int:
    by_layer = host.nonlinear_sites_by_layer()
    wanted = set(sites)
    n = 0

    for layer, site, window in assignment.items():
        if site not in wanted:
            continue
        entry = by_layer.get(layer)
        if entry is None:
            raise KeyError(
                f"assignment names layer {layer} but the patched model has layers "
                f"{sorted(by_layer)}"
            )
        _reset(entry[site], window, layer, site)
        n += 1

    if n == 0:
        raise ValueError(
            f"assignment applied 0 windows (assignment has {len(assignment)} entries, "
            f"sites filter is {sorted(wanted)})"
        )
    return n
