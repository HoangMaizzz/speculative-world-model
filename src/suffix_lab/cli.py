import argparse
import json
from pathlib import Path
from .assets import ROOT


def configuration(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.smoke:
        config.update(prompts_per_dataset=5, prefixes_per_prompt=3, reference_tokens=256,
                      test2_events=24, oracle_subset_limit=4, bootstrap_samples=200)
    if not 0 < config["prefixes_per_prompt"] <= 5 or config["proposal_length"] != 32:
        raise ValueError("This protocol requires proposal32 and 1–5 prefixes")
    if (config["physical_block_size"], config["subblock_size"]) != (32, 8):
        raise ValueError("This protocol requires physical32 / active8")
    if set(config["redrafters"]) != {"dllm_top1", "ar_greedy"} or len(config["redrafters"]) != 2:
        raise ValueError("Both redrafters must run exactly once")
    if config["conditioning_anchor_count"] < 1 or config["conditioning_probe_tokens"] < 1:
        raise ValueError("Conditioning needs positive anchor count and probe horizon")
    if config["test2_events"] < 2:
        raise ValueError("Test2 needs a budget for both cohorts")
    if not config["conditioning_seeds"] or len(set(config["conditioning_seeds"])) != len(config["conditioning_seeds"]):
        raise ValueError("Conditioning seeds must be nonempty and unique")
    return config


def main():
    parser = argparse.ArgumentParser(description="Offline suffix diagnostics; no online speedup claims")
    parser.add_argument("command", choices=["prepare", "preflight", "run", "analyze", "pack", "restore", "import-checkpoint"])
    parser.add_argument("--config", default=str(ROOT / "configs/math_gsm8k.json"))
    parser.add_argument("--assets", default="assets/pilot")
    parser.add_argument("--output", default="outputs/pilot")
    parser.add_argument("--smoke", action="store_true", help="Separate 5+5 smoke identity; use separate output/assets")
    parser.add_argument("--stage", choices=["all", "references", "test1", "test2"], default="all")
    parser.add_argument("--archive")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--revision")
    args = parser.parse_args()
    config = configuration(args)
    if args.command == "import-checkpoint":
        from .migration import import_checkpoint
        if not args.checkpoint_root or not args.revision:
            parser.error("import-checkpoint requires --checkpoint-root and --revision")
        import_checkpoint(args.checkpoint_root, args.output, config, args.revision)
    elif args.command == "prepare":
        from .assets import prepare
        prepare(config, args.assets)
    elif args.command == "restore":
        from .experiment import restore
        if not args.archive:
            parser.error("restore requires --archive")
        restore(args.archive, args.output)
    elif args.command == "pack":
        from .experiment import pack
        print(pack(args.output))
    elif args.command == "analyze":
        from .analysis import analyze
        print(json.dumps(analyze(args.output, config["bootstrap_samples"]), indent=2))
    else:
        from .assets import read_assets
        from .runtime import preflight
        manifest = read_assets(args.assets, config)
        if args.command == "preflight":
            print(json.dumps(preflight(manifest, config, args.output), indent=2))
        else:
            from .experiment import Runner
            result = Runner(config, manifest, args.output, args.stage).run()
            print("Run status:", result)
            if result != "complete":
                raise SystemExit(75)


if __name__ == "__main__":
    main()
