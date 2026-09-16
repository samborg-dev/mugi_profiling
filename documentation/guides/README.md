# Documentation

Welcome! This folder documents the per-layer LUT window search — the part of
`mugi_profiling` that decides, automatically, where each of Llama-2-7B's 32 layers should
place its lookup-table exponent window.

These guides are written to be read by someone who works on Mugi but not on this code.
If you are here because you want to consume the search's output from the runtime side,
start with [Inputs and Outputs](INPUTS_AND_OUTPUTS.md) and
[Runtime Integration](RUNTIME_INTEGRATION.md) — between them they answer the whole
question.

---

## The Guides

| Guide | Read this if you want to know... |
|---|---|
| [Inputs and Outputs](INPUTS_AND_OUTPUTS.md) | What the framework takes in and what it writes out, field by field |
| [Runtime Integration](RUNTIME_INTEGRATION.md) | How the output reaches the runtime, and whether a compiler stack is needed |
| [The Window Model](WINDOW_MODEL.md) | What a "window" actually is — there are three of them, and only one is searched |
| [The Search](SEARCH.md) | How candidates are generated, scored, and frozen layer by layer |

---

## The Short Version

If you read nothing else:

- **Input** is the same three YAML configs the existing pipeline already takes.
- **Output** is a per-layer window assignment plus a full audit trail of every candidate
  evaluated.
- **No new compiler stack is needed.** The output lands in a schema the runtime already
  reads — `create_nonlinear_config` has been emitting that shape all along. What changed is
  how the numbers are chosen, not what a window is or how it is expressed.
- The one gap is that the conversion currently runs in-process only; nothing yet writes the
  assignment back out as a `nonlinear_config.yaml` on disk. That is about fifteen lines of
  work, described at the end of [Runtime Integration](RUNTIME_INTEGRATION.md).

---

## About the Diagrams

Three of the four guides embed a D2 diagram — [The Search](SEARCH.md) is prose and code
only. The `.d2` file is the source; the `.svg` is what the markdown displays.

If you edit a source, re-render it:

```bash
cd documentation/guides
d2 window_search_io.d2           window_search_io.svg
d2 window_search_runtime.d2      window_search_runtime.svg
d2 window_search_window_model.d2 window_search_window_model.svg
```

The parent `documentation/` folder holds additional D2 diagrams from earlier work. They are
not referenced by this series.

---

## A Note on Scope

These guides describe what the code **does today**. Where behaviour is surprising, or where
a name suggests something other than what the code performs, the guides say so rather than
describing the version we wish existed.

Numbers come from two places, and the guides distinguish them. Perplexity figures for
Llama-2-7B are from the ASPLOS '26 paper, with the figure or section named so you can check
them. Timings are from our own runs on Delta and are labelled as such — they will differ on
other hardware.
