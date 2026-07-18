import argparse

import yaml

from profiling_api.config import ProfileConfig


def build_config(args) -> ProfileConfig:
    model_config = yaml.safe_load(open(args.model_config))
    nonlinear_config = yaml.safe_load(open(args.nonlinear_config))
    parameter_config = yaml.safe_load(open(args.parameter_config))

    overrides = {}
    if args.calibration is not None:
        overrides["window_calibration"] = args.calibration
    if args.array_config is not None:
        overrides["array_config"] = args.array_config
    if args.variant is not None:
        overrides["variant"] = args.variant
    if args.max_seq_len is not None:
        overrides["max_seq_len"] = args.max_seq_len
    if args.prefill_seq_len is not None:
        overrides["prefill_seq_len"] = args.prefill_seq_len
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir

    return ProfileConfig.from_configs(model_config, nonlinear_config, parameter_config, **overrides)


def main():
    parser = argparse.ArgumentParser(description="Run the Mugi profiling pipeline end to end.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--nonlinear_config", required=True)
    parser.add_argument("--parameter_config", required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--array_config", default=None)
    parser.add_argument("--variant", default=None, choices=["lut", "vlp"])
    parser.add_argument("--max_seq_len", type=int, default=None)
    parser.add_argument("--prefill_seq_len", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    cfg = build_config(args)

    from profiling_api import ProfilingPipeline

    result = ProfilingPipeline(cfg).run()
    print("profiling complete:")
    for key, path in result.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
