"""Fresh-session orchestration for either smoke or pilot; no smoke-file dependency."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from suffix_lab.cli import configuration
from suffix_lab.store import atomic_json

ROOT = Path(__file__).resolve().parents[1]


def matching_checkpoint(input_root, config, revision):
    matches = []
    for path in sorted(Path(input_root).rglob("suffix_checkpoint.zip")):
        with zipfile.ZipFile(path) as archive:
            if "run-manifest.json" not in archive.namelist():
                continue
            manifest = json.loads(archive.read("run-manifest.json"))
        if manifest.get("config") == config and manifest.get("environment", {}).get("commit") == revision:
            matches.append(path)
    if len(matches) > 1:
        raise ValueError("Multiple matching checkpoints: attach only the desired checkpoint before running.")
    return matches[0] if matches else None


def execute(mode, revision, assets, output, input_root="/kaggle/input", run=subprocess.run):
    if mode not in ("smoke", "pilot"):
        raise ValueError(mode)
    output, assets = Path(output), Path(assets)
    output.mkdir(parents=True, exist_ok=True)
    cfg = configuration(argparse.Namespace(config=str(ROOT/"configs/math_gsm8k.json"), smoke=mode == "smoke"))
    phase = "restore"
    status, exit_code = "failed", None
    started = time.monotonic()
    cleanup_errors = []

    def cli(command, extra=(), allowed=(0,)):
        nonlocal phase
        phase = command
        args = [sys.executable, "-u", "-m", "suffix_lab.cli", command,
                "--assets", str(assets), "--output", str(output)]
        if mode == "smoke":
            args.append("--smoke")
        args.extend(extra)
        print(f"[{datetime.now(timezone.utc).isoformat()}] {mode}: {command}", flush=True)
        result = run(args, cwd=ROOT)
        if result.returncode not in allowed:
            raise subprocess.CalledProcessError(result.returncode, args)
        return result.returncode

    try:
        if not (output/"run.sqlite").exists():
            unpacked = os.environ.get("SUFFIX_RESUME_ROOT")
            if unpacked:
                cli("import-checkpoint", ("--checkpoint-root", unpacked, "--revision", revision))
            checkpoint = None if unpacked else matching_checkpoint(input_root, cfg, revision)
            if checkpoint:
                # restore requires an empty destination; no status files are written yet.
                if any(output.iterdir()):
                    raise ValueError("Output contains files but no live database; preserve them and choose an empty output to restore.")
                cli("restore", ("--archive", str(checkpoint)))
                print("Restored matching checkpoint:", checkpoint, flush=True)
        cli("prepare")
        manifest = json.loads((assets/"manifest.json").read_text())
        phase = "unit_checks"
        test_env = dict(os.environ)
        # Unit orchestration fixtures must not inherit the real migration input.
        test_env.pop("SUFFIX_RESUME_ROOT", None)
        test_env["SUFFIX_UPSTREAM_DIR"] = manifest["paths"]["Efficient-Large-Model/Fast_dLLM_v2_1.5B"]
        tests = [sys.executable, "-u", "-m", "unittest", "discover", "-s", "tests", "-v"]
        checked = run(tests, cwd=ROOT, env=test_env)
        if checked.returncode:
            raise subprocess.CalledProcessError(checked.returncode, tests)
        # Both modes run real-model checks. Pilot does not depend on an interactive
        # smoke status: Save Version starts its own clean process/session.
        cli("preflight")
        exit_code = cli("run", allowed=(0, 75))
        status = "partial" if exit_code == 75 else "complete"
    finally:
        failed_phase = phase
        if (output/"run.sqlite").exists():
            try:
                cli("analyze")
            except Exception as error:
                cleanup_errors.append("analysis: "+str(error))
        if cleanup_errors:
            status = "failed"
        payload = {"mode": mode, "status": status, "last_experiment_phase": failed_phase,
                   "experiment_exit_code": exit_code, "code_revision": revision,
                   "total_seconds_including_prepare_preflight": time.monotonic()-started,
                   "cleanup_errors": cleanup_errors}
        atomic_json(output/"execution_status.json", payload)
        try:
            cli("pack")
        except Exception as error:
            cleanup_errors.append("pack: "+str(error))
            payload.update(status="failed", cleanup_errors=cleanup_errors)
            atomic_json(output/"execution_status.json", payload)
            raise
        print(json.dumps(payload, indent=2), flush=True)
    if cleanup_errors:
        raise RuntimeError("; ".join(cleanup_errors))
    if status == "partial":
        print("PARTIAL: soft run budget reached. ZIP is saved. Attach it as input to a new Save Version and rerun the same pilot cell.", flush=True)
    else:
        print("COMPLETE:", mode, flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot"], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if actual != args.revision:
        raise ValueError("Checkout revision differs from the pinned launcher")
    execute(args.mode, args.revision, args.assets, args.output)


if __name__ == "__main__":
    main()
