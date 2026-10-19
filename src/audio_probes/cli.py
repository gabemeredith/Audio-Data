"""Command line entry point.

    uv run python -m audio_probes <stage> --property <name> [--condition <c>] [key.sub=value ...]

Stages: data, embed, probe, llm, report, inspect, all, show-config.
"""

from __future__ import annotations

import argparse
import importlib

import yaml

from audio_probes.config import load_config
from audio_probes.properties import PROPERTY_NAMES

# stage name -> "module:function". Each function takes (cfg, condition) except data,
# which runs once per property because conditions are applied after generation.
STAGES = {
    "data": "audio_probes.data:run",
    "embed": "audio_probes.embed:run",
    "probe": "audio_probes.probes:run",
    "llm": "audio_probes.llm_eval:run",
    "report": "audio_probes.report:run",
}
PIPELINE = ["data", "embed", "probe", "llm", "report"]


def _stage_fn(stage: str):
    module_name, fn_name = STAGES[stage].split(":")
    return getattr(importlib.import_module(module_name), fn_name)


def run_stage(stage: str, cfg: dict, conditions: list[str]) -> None:
    fn = _stage_fn(stage)
    if stage == "data":
        fn(cfg)
        return
    for condition in conditions:
        fn(cfg, condition)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="audio_probes", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=[*STAGES, "all", "inspect", "show-config"])
    parser.add_argument("--property", "-p", default="all",
                        help=f"one of {PROPERTY_NAMES} or 'all'")
    parser.add_argument("--condition", "-c", default=None,
                        help="limit to one condition (default: all conditions in the config)")
    parser.add_argument("--clip", help="clip id, for the inspect stage")
    parser.add_argument("--run", help="results run hash, for inspect (default: latest)")
    parser.add_argument("overrides", nargs="*", help="config overrides like probe.alpha=10")
    args = parser.parse_args(argv)

    properties = PROPERTY_NAMES if args.property == "all" else [args.property]
    for name in properties:
        cfg = load_config(name, args.overrides)
        conditions = [args.condition] if args.condition else list(cfg["conditions"])
        if args.stage == "show-config":
            print(yaml.safe_dump(cfg, sort_keys=False))
        elif args.stage == "inspect":
            from audio_probes.inspect_view import run as inspect_run
            if not args.clip:
                parser.error("inspect needs --clip <id>")
            for condition in conditions:
                inspect_run(cfg, condition, args.clip, args.run)
        elif args.stage == "all":
            for stage in PIPELINE:
                run_stage(stage, cfg, conditions)
        else:
            run_stage(args.stage, cfg, conditions)
