"""OM-LEU (PO-LEU) with an open-weight generator, on Modal.

One command runs the paper's protocol on ONE dataset:

    prepare  ->  per-seed [vLLM server + OM-LEU training + baseline
                 leaderboard on the same records]  ->  paired significance

The open-weight LLM is served *inside* the GPU container by vLLM's
OpenAI-compatible endpoint; ``scripts/run_dataset.py`` talks to it through
``LLM_PROVIDER=openai_compatible`` exactly as it talks to Anthropic /
Gemini / OpenAI, so the OM-LEU code path is unchanged and the model id is
folded into the outcomes-cache key.

Datasets
--------
``--dataset swissmetro`` (default) — public Biogeme SP mode-choice data;
downloaded from EPFL automatically when ``--raw`` is omitted.
``--dataset expedia_rectour`` — Expedia RecTour lodging searches; the raw
file is request-only, pass it with ``--raw``.

Usage
-----

    # Swissmetro, paper protocol (3 seeds), Llama-3.3-70B FP8 on one H200
    modal run modal_app/omleu_eval.py --dataset swissmetro --seeds 7,11,13 \\
        --n-customers 200 --n-epochs 20 --llm-baselines

    # Expedia RecTour
    modal run modal_app/omleu_eval.py --dataset expedia_rectour \\
        --raw ~/Downloads/rectour_searches.csv --amenities ~/Downloads/property_amenities.csv \\
        --seeds 7,11,13 --n-customers 100

Knobs
-----
``--model`` any HF model id vLLM can serve (default: RedHatAI FP8 build of
Llama-3.3-70B-Instruct, ungated, fits one H200 / two H100s). ``--gpu`` is
read from ``OMLEU_GPU`` (default ``H200``). ``--llm-baselines`` also runs
the ZeroShot / FewShot-ICL rankers with the same open-weight model;
``--symbolic-llm-baselines`` adds LLM-SR / LaSR (iterative, slow).
Results are downloaded to ``<dataset>/results/<run_tag>/``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import modal

APP_NAME = "omleu-eval"
REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = "/root/repo"

DEFAULT_MODEL = os.environ.get(
    "OMLEU_GEN_MODEL", "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic"
)
DEFAULT_GPU = os.environ.get("OMLEU_GPU", "H200")
VLLM_PORT = 8000

# Per-dataset wiring: prepare script, run_dataset adapter, prompt family.
DATASETS: dict[str, dict] = {
    "swissmetro": {
        "prepare_script": "scripts/prepare_swissmetro.py",
        "adapter": "swissmetro",
        "prompt_version": "v6_travel_anchored",
        "yaml": "configs/datasets/swissmetro.yaml",
        "extras_block": "swissmetro",
        "min_events": 5,
        "public_url": "https://transp-or.epfl.ch/data/swissmetro.dat",
        "raw_name": "swissmetro.dat",
    },
    "expedia_rectour": {
        "prepare_script": "scripts/prepare_expedia_rectour.py",
        "adapter": "expedia_rectour",
        "prompt_version": "v5_hotel_anchored",
        "yaml": "configs/datasets/expedia_rectour.yaml",
        "extras_block": "expedia_rectour",
        "min_events": 3,
        "public_url": None,
        "raw_name": "rectour_searches.csv",
    },
    # Optima: mostly one loop per respondent -> between-customer split.
    "optima": {
        "prepare_script": "scripts/prepare_optima.py",
        "adapter": "optima",
        "prompt_version": "v7_modechoice_anchored",
        "yaml": "configs/datasets/optima.yaml",
        "extras_block": "modechoice",
        "min_events": 1,
        "split_mode": "cold_start",
        "public_url": None,
        "raw_name": "optima.dat",
    },
    # LPMC: timestamped trips per person -> the paper's temporal split.
    "lpmc": {
        "prepare_script": "scripts/prepare_lpmc.py",
        "adapter": "lpmc",
        "prompt_version": "v7_modechoice_anchored",
        "yaml": "configs/datasets/lpmc.yaml",
        "extras_block": "modechoice",
        "min_events": 5,
        "public_url": None,
        "raw_name": "lpmc.dat",
    },
}

DATA_VOL_NAME = "omleu-expedia-data"
RESULTS_VOL_NAME = "omleu-expedia-results"
HF_VOL_NAME = "omleu-hf-cache"

data_vol = modal.Volume.from_name(DATA_VOL_NAME, create_if_missing=True)
results_vol = modal.Volume.from_name(RESULTS_VOL_NAME, create_if_missing=True)
hf_vol = modal.Volume.from_name(HF_VOL_NAME, create_if_missing=True)

DATA_DIR = "/data"
RESULTS_DIR = "/results"
CACHE_DIR = "/root/.cache"  # HF weights + FlashInfer JIT cache persist here
HF_DIR = CACHE_DIR  # HF_HOME; the hub cache lands at /root/.cache/hub on the volume

# CUDA *devel* base: vLLM 0.28 ships cu13 wheels and FlashInfer JIT-compiles
# kernels at first use, which needs nvcc under /usr/local/cuda.
CUDA_IMAGE = os.environ.get("OMLEU_CUDA_IMAGE", "nvidia/cuda:13.0.1-devel-ubuntu22.04")

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "curl")
    .pip_install(
        "vllm==0.28.0",
        "huggingface_hub[hf_transfer]>=0.30",
    )
    .pip_install(
        "numpy>=1.24",
        "pandas>=2.0,<3",
        "pyyaml>=6.0",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "statsmodels>=0.14",
        "sentence-transformers>=2.7",
        "tqdm>=4.66",
        "openai>=1.0",
        "python-dotenv>=1.0",
        # Bayesian-ARD baseline (NumPyro NUTS on CPU).
        "jax[cpu]>=0.4",
        "numpyro>=0.15",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": HF_DIR,
            "CUDA_HOME": "/usr/local/cuda",
            # scripts/run_baselines.py and paired_significance.py import
            # ``src`` without patching sys.path (they expect ``python -m``).
            "PYTHONPATH": REMOTE_REPO,
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(REPO_ROOT / "src", f"{REMOTE_REPO}/src")
    .add_local_dir(REPO_ROOT / "scripts", f"{REMOTE_REPO}/scripts")
    .add_local_dir(REPO_ROOT / "configs", f"{REMOTE_REPO}/configs")
)

app = modal.App(APP_NAME)


# --------------------------------------------------------------------------- #
# Helpers (run inside containers)
# --------------------------------------------------------------------------- #


def _sh(cmd: list[str], *, env: dict | None = None, cwd: str = REMOTE_REPO,
        log_path: str | None = None) -> int:
    """Run a command, streaming output (and teeing to ``log_path``)."""
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    log_fh = open(log_path, "a") if log_path else None
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=full_env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        if log_fh:
            log_fh.write(line)
    proc.wait()
    if log_fh:
        log_fh.close()
    return int(proc.returncode)


def _resolved_dataset_yaml(dataset: str, prepared_dir: str, out_path: str) -> str:
    """Point the dataset YAML's file paths at ``prepared_dir``."""
    import yaml

    spec = DATASETS[dataset]
    src = Path(REMOTE_REPO) / spec["yaml"]
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    d = Path(prepared_dir)
    doc["dataset"]["events"]["path"] = str(d / "events.csv")
    doc["dataset"]["persons"]["path"] = str(d / "persons.csv")
    ex = doc.setdefault(spec["extras_block"], {})
    for key in list(ex.keys()):
        if key.endswith("_path"):
            ex[key] = str(d / Path(str(ex[key])).name)
    ex.setdefault("impressions_path", str(d / "impressions.csv"))
    ex.setdefault("properties_path", str(d / "properties.csv"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out_path


def _start_vllm(model_id: str, *, max_model_len: int, gpu_mem: float,
                tensor_parallel: int, log_path: str) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", model_id,
        "--port", str(VLLM_PORT),
        "--host", "127.0.0.1",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--tensor-parallel-size", str(tensor_parallel),
        "--max-num-seqs", "128",
    ]
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    log_fh = open(log_path, "a")
    return subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)


def _wait_for_vllm(proc: subprocess.Popen, *, timeout_s: int = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vLLM exited early with code {proc.returncode}; see vllm.log"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5
            ) as r:
                if r.status == 200:
                    print(f"vLLM ready after {time.time() - t0:.0f}s", flush=True)
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become healthy in time")


def _smoke_vllm(model_id: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=f"http://127.0.0.1:{VLLM_PORT}/v1", api_key="not-needed")
    r = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=5, temperature=0.0,
    )
    return (r.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Remote functions
# --------------------------------------------------------------------------- #


@app.function(
    image=image,
    volumes={DATA_DIR: data_vol},
    timeout=60 * 60 * 3,
    cpu=8.0,
    memory=32768,
)
def prepare(
    dataset: str,
    raw_rel: str,
    prepared_tag: str,
    extra_args: list[str],
) -> dict:
    """Raw file on the data volume -> prepared CSVs on the same volume."""
    spec = DATASETS[dataset]
    data_vol.reload()
    raw = Path(DATA_DIR) / raw_rel
    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not on volume {DATA_VOL_NAME}. Upload it first:\n"
            f"  modal volume put {DATA_VOL_NAME} <local path> {raw_rel}"
        )
    out_dir = Path(DATA_DIR) / "prepared" / prepared_tag
    cmd = [
        sys.executable, spec["prepare_script"],
        "--raw", str(raw), "--out", str(out_dir), *extra_args,
    ]
    rc = _sh(cmd)
    if rc != 0:
        raise RuntimeError(f"prepare failed with exit {rc}")
    data_vol.commit()
    summary = json.loads((out_dir / "prepare_summary.json").read_text())
    return {"prepared_dir": str(out_dir), "summary": summary}


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={DATA_DIR: data_vol, RESULTS_DIR: results_vol, CACHE_DIR: hf_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60 * 8,
    cpu=8.0,
    memory=65536,
)
def run_seed(
    dataset: str,
    prepared_tag: str,
    run_tag: str,
    seed: int,
    n_customers: int,
    n_epochs: int,
    batch_size: int,
    model_id: str,
    max_concurrency: int = 64,
    max_model_len: int = 4096,
    gpu_mem: float = 0.88,
    tensor_parallel: int = 1,
    min_events_per_customer: int | None = None,
    llm_baselines: bool = False,
    symbolic_llm_baselines: bool = False,
    max_tokens: int = 220,
    baselines_only: bool = False,
    train_config: str = "configs/default.yaml",
) -> dict:
    """vLLM server + OM-LEU training + baseline leaderboard for one seed.

    ``baselines_only=True`` reuses an existing ``omleu/records.pkl`` +
    ``test_logits.npz`` for this run_tag/seed and re-runs only the
    baseline leaderboard (e.g. after a baseline-code fix)."""
    spec = DATASETS[dataset]
    if min_events_per_customer is None:
        min_events_per_customer = int(spec["min_events"])
    data_vol.reload()
    prepared_dir = Path(DATA_DIR) / "prepared" / prepared_tag
    if not (prepared_dir / "events.csv").exists():
        raise FileNotFoundError(f"prepared data missing: {prepared_dir}")

    run_dir = Path(RESULTS_DIR) / run_tag / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    omleu_dir = run_dir / "omleu"
    base_dir = run_dir / "baselines"
    cache_dir = run_dir / "cache"
    for d in (omleu_dir, base_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "dataset": dataset, "prepared_tag": prepared_tag, "seed": seed,
        "n_customers": n_customers, "n_epochs": n_epochs, "batch_size": batch_size,
        "model_id": model_id, "max_concurrency": max_concurrency,
        "min_events_per_customer": min_events_per_customer,
        "llm_baselines": llm_baselines,
        "symbolic_llm_baselines": symbolic_llm_baselines, "gpu": DEFAULT_GPU,
        "prompt_version": spec["prompt_version"], "K": 5, "max_tokens": max_tokens,
        "train_config": train_config,
    }, indent=2))

    yaml_path = _resolved_dataset_yaml(
        dataset, str(prepared_dir), str(run_dir / f"{dataset}.yaml")
    )

    vllm_proc = _start_vllm(
        model_id, max_model_len=max_model_len, gpu_mem=gpu_mem,
        tensor_parallel=tensor_parallel, log_path=str(run_dir / "vllm.log"),
    )
    try:
        _wait_for_vllm(vllm_proc)
        print("vLLM smoke:", _smoke_vllm(model_id), flush=True)
        hf_vol.commit()  # persist freshly downloaded weights for later seeds

        env = {
            "LLM_PROVIDER": "openai_compatible",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{VLLM_PORT}/v1",
            "OPENAI_MODEL": model_id,
            "OPENAI_API_KEY": "not-needed",
            "MAX_CONCURRENT_LLM_CALLS": str(max_concurrency),
            "OUTCOMES_MAX_TOKENS": str(max_tokens),
            "OUTCOMES_CACHE_PATH": str(cache_dir / "outcomes.sqlite"),
            "EMBEDDINGS_CACHE_PATH": str(cache_dir / "embeddings.sqlite"),
        }
        # ---- OM-LEU ------------------------------------------------------
        t0 = time.time()
        have_omleu = (omleu_dir / "records.pkl").exists() and (omleu_dir / "test_logits.npz").exists()
        if baselines_only and have_omleu:
            print(f"baselines_only: reusing {omleu_dir}", flush=True)
            omleu_seconds = 0.0
        else:
            if baselines_only:
                raise FileNotFoundError(
                    f"baselines_only requested but {omleu_dir} has no records.pkl/test_logits.npz"
                )
            rc = _sh([
                sys.executable, "scripts/run_dataset.py",
                "--adapter", spec["adapter"],
                "--dataset-config", yaml_path,
                "--n-customers", str(n_customers), "--seed", str(seed),
                "--K", "5", "--prompt-version-cascade", spec["prompt_version"],
                "--min-events-per-customer", str(min_events_per_customer),
                "--split-mode", str(spec.get("split_mode", "temporal")),
                "--n-epochs", str(n_epochs), "--batch-size", str(batch_size),
                "--config", train_config,
                "--output-dir", str(omleu_dir),
            ], env=env, log_path=str(run_dir / "omleu.log"))
            results_vol.commit()
            if rc != 0:
                raise RuntimeError(f"run_dataset.py exit {rc}; see {run_dir}/omleu.log")
            omleu_seconds = time.time() - t0

        # ---- baselines on the SAME records -------------------------------
        base_env = dict(env)
        if llm_baselines:
            base_env["OPENAI_COMPAT_MODEL"] = model_id
            suffix = model_id.split("/")[-1].replace(":", "-").replace(" ", "-")
            base_env["LLM_SWEEP"] = suffix
            if not symbolic_llm_baselines:
                base_env["LLM_BASELINE_SKIP"] = "LLM-SR,LaSR"
        else:
            base_env["LLM_BASELINE_SKIP"] = "ZeroShot,FewShot-ICL,LLM-SR,LaSR"
        t1 = time.time()
        rc = _sh([
            sys.executable, "scripts/run_baselines.py",
            "--records-from", str(omleu_dir / "records.pkl"),
            "--external-logits", f"OM-LEU={omleu_dir / 'test_logits.npz'}",
            "--output-dir", str(base_dir),
            "--tag", f"main_seed{seed}",
        ], env=base_env, log_path=str(run_dir / "baselines.log"))
        results_vol.commit()
        if rc != 0:
            raise RuntimeError(f"run_baselines.py exit {rc}; see {run_dir}/baselines.log")
        baseline_seconds = time.time() - t1
    finally:
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    metrics_test = json.loads((omleu_dir / "metrics_test.json").read_text())
    lb_txt = base_dir / f"baselines_leaderboard_main_seed{seed}.txt"
    leaderboard = lb_txt.read_text() if lb_txt.exists() else ""
    out = {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "seed": seed,
        "model_id": model_id,
        "omleu_seconds": round(omleu_seconds, 1),
        "baseline_seconds": round(baseline_seconds, 1),
        "metrics_test": metrics_test,
        "leaderboard": leaderboard,
    }
    (run_dir / "seed_summary.json").write_text(json.dumps(out, indent=2))
    results_vol.commit()
    return out


@app.function(
    image=image,
    volumes={RESULTS_DIR: results_vol},
    timeout=60 * 60,
    cpu=4.0,
)
def aggregate(run_tag: str, seeds: list[int]) -> dict:
    """Mean ± sd across seeds + paired significance vs every baseline."""
    results_vol.reload()
    run_root = Path(RESULTS_DIR) / run_tag
    agg_dir = run_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    lb_dir = agg_dir / "leaderboards"
    lb_dir.mkdir(exist_ok=True)
    rows_by_seed: dict[int, list[dict]] = {}
    for s in seeds:
        src = run_root / f"seed_{s}" / "baselines" / f"baselines_leaderboard_main_seed{s}.json"
        if not src.exists():
            print(f"WARNING: missing {src}")
            continue
        (lb_dir / src.name).write_bytes(src.read_bytes())
        payload = json.loads(src.read_text())
        rows = payload["rows"] if isinstance(payload, dict) and "rows" in payload else payload
        rows_by_seed[s] = rows

    import numpy as np

    by_name: dict[str, dict[str, list[float]]] = {}
    for s, rows in rows_by_seed.items():
        for r in rows:
            if r.get("status", "ok") != "ok":
                continue
            d = by_name.setdefault(str(r["name"]), {})
            for k in ("top1", "top3", "top5", "mrr", "test_nll", "brier", "ece"):
                v = r.get(k)
                if v is not None:
                    d.setdefault(k, []).append(float(v))
    table_rows = []
    for name, d in by_name.items():
        row = {"name": name, "n_seeds": len(d.get("test_nll", []))}
        for k, vals in d.items():
            row[f"{k}_mean"] = float(np.mean(vals))
            row[f"{k}_sd"] = float(np.std(vals, ddof=0))
        table_rows.append(row)
    table_rows.sort(key=lambda r: r.get("test_nll_mean", 1e9))
    (agg_dir / "summary_over_seeds.json").write_text(json.dumps(table_rows, indent=2))

    lines = [f"{'method':40s} {'Top-1':>12s} {'Top-3':>12s} {'Top-5':>12s} {'MRR':>12s} {'NLL':>14s} {'Brier':>14s}  n"]
    for r in table_rows:
        def f(k, pct=False, w=12):
            m, sd = r.get(f"{k}_mean"), r.get(f"{k}_sd")
            if m is None:
                return " " * w
            return (f"{100*m:5.1f}±{100*sd:4.1f}%" if pct else f"{m:.4f}±{sd:.4f}").rjust(w)
        lines.append(
            f"{r['name']:40s} {f('top1', True)} {f('top3', True)} {f('top5', True)} "
            f"{f('mrr')} {f('test_nll', w=14)} {f('brier', w=14)}  {r['n_seeds']}"
        )
    table = "\n".join(lines)
    (agg_dir / "summary_over_seeds.txt").write_text(table)
    print(table, flush=True)

    sig_out = ""
    if rows_by_seed:
        sig_log = agg_dir / "significance.log"
        if sig_log.exists():
            sig_log.unlink()
        rc = _sh([
            sys.executable, "scripts/paired_significance.py",
            "--input-dir", str(lb_dir), "--tag-pattern", "main_seed*",
            "--baseline-of-interest", "OM-LEU",
            "--output-dir", str(agg_dir / "significance"),
        ], log_path=str(sig_log))
        md = agg_dir / "significance" / "pairwise_vs_OM-LEU.md"
        sig_out = md.read_text() if md.exists() else sig_log.read_text()[-4000:]
        if rc != 0:
            print(f"WARNING: paired_significance.py exit {rc} (see significance.log)")
    results_vol.commit()
    return {"agg_dir": str(agg_dir), "table": table, "significance": sig_out}


# --------------------------------------------------------------------------- #
# Local entrypoint
# --------------------------------------------------------------------------- #


def _upload(local: Path, remote_rel: str) -> None:
    print(f"uploading {local} -> {DATA_VOL_NAME}:{remote_rel}", flush=True)
    with data_vol.batch_upload(force=True) as batch:
        batch.put_file(str(local), remote_rel)


def _on_volume(remote_rel: str) -> bool:
    try:
        parent = str(Path(remote_rel).parent)
        return remote_rel in {e.path for e in data_vol.listdir(parent)}
    except Exception:
        return False


@app.local_entrypoint()
def main(
    dataset: str = "swissmetro",
    raw: str = "",
    amenities: str = "",
    prepared_tag: str = "",
    run_tag: str = "",
    seeds: str = "7,11,13",
    n_customers: int = 200,
    n_epochs: int = 20,
    batch_size: int = 32,
    model: str = DEFAULT_MODEL,
    max_concurrency: int = 64,
    tensor_parallel: int = 1,
    gpu_mem: float = 0.88,
    slate_size: int = 10,
    min_events: int = 0,
    max_users: int = 3000,
    label_mode: str = "click_or_book",
    llm_baselines: bool = False,
    symbolic_llm_baselines: bool = False,
    skip_prepare: bool = False,
    baselines_only: bool = False,
    train_config: str = "configs/default.yaml",
    download_to: str = "",
):
    """See module docstring. ``--raw`` is a local file (uploaded for you) or
    a path already on the data volume; Swissmetro is fetched from EPFL
    when ``--raw`` is omitted."""
    if dataset not in DATASETS:
        raise SystemExit(f"--dataset must be one of {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    if not seed_list:
        raise SystemExit("--seeds must list at least one integer")
    min_events_eff = int(min_events) if min_events > 0 else int(spec["min_events"])
    prepared_tag = prepared_tag or dataset
    if baselines_only:
        if not run_tag:
            raise SystemExit("--baselines-only needs --run-tag of an existing run")
        skip_prepare = True
    run_tag = run_tag or f"{dataset}_{time.strftime('%Y%m%d_%H%M%S')}"

    extra_args: list[str] = []
    raw_rel = raw
    if not skip_prepare:
        if not raw and spec["public_url"]:
            local = REPO_ROOT / "tmp" / dataset / spec["raw_name"]
            local.parent.mkdir(parents=True, exist_ok=True)
            if not local.exists():
                print(f"downloading {spec['public_url']} -> {local}", flush=True)
                urllib.request.urlretrieve(spec["public_url"], local)
            raw = str(local)
        staged = REPO_ROOT / dataset / "data" / "raw" / spec["raw_name"]
        if not raw and staged.exists():
            raw = str(staged)  # <dataset>/data/raw/<raw_name> is the conventional spot
        if not raw:
            raise SystemExit(
                f"--raw is required for {dataset} (local file, or a path on the "
                f"'{DATA_VOL_NAME}' volume)."
            )
        if Path(raw).expanduser().exists():
            raw_rel = f"raw/{dataset}/{Path(raw).name}"
            _upload(Path(raw).expanduser(), raw_rel)
        elif not _on_volume(raw):
            raise SystemExit(
                f"--raw {raw!r} is neither a local file nor a file on the "
                f"'{DATA_VOL_NAME}' volume."
            )
        if dataset == "expedia_rectour":
            extra_args += ["--J", str(slate_size), "--min-events", str(min_events_eff),
                           "--max-users", str(max_users), "--seed", "42",
                           "--label-mode", label_mode]
            if amenities:
                am_rel = amenities
                if Path(amenities).expanduser().exists():
                    am_rel = f"raw/{dataset}/{Path(amenities).name}"
                    _upload(Path(amenities).expanduser(), am_rel)
                extra_args += ["--amenities", f"{DATA_DIR}/{am_rel}"]
        prep = prepare.remote(dataset, raw_rel, prepared_tag, extra_args)
        print("prepare:", json.dumps(prep["summary"], indent=2)[:2500], flush=True)

    print(f"launching {len(seed_list)} seed run(s) on {DEFAULT_GPU} with {model}", flush=True)
    handles = [
        run_seed.spawn(
            dataset, prepared_tag, run_tag, s, n_customers, n_epochs, batch_size, model,
            max_concurrency=max_concurrency, tensor_parallel=tensor_parallel,
            gpu_mem=gpu_mem, min_events_per_customer=min_events_eff,
            llm_baselines=llm_baselines, symbolic_llm_baselines=symbolic_llm_baselines,
            baselines_only=baselines_only, train_config=train_config,
        )
        for s in seed_list
    ]
    for h in handles:
        r = h.get()
        print(f"\n=== seed {r['seed']} done: OM-LEU {r['omleu_seconds']}s, "
              f"baselines {r['baseline_seconds']}s ===")
        print(json.dumps(r["metrics_test"], indent=2)[:1500])
        print(r["leaderboard"][:4000])

    agg = aggregate.remote(run_tag, seed_list)
    print("\n=== mean ± sd over seeds ===")
    print(agg["table"])
    if agg.get("significance"):
        print("\n=== paired significance vs OM-LEU ===")
        print(agg["significance"][:4000])

    # Results land under the dataset's own directory: <dataset>/results/<run_tag>.
    dest = REPO_ROOT / (download_to or f"{dataset}/results") / run_tag
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\ndownloading results to {dest}", flush=True)
    rc = subprocess.call([
        "modal", "volume", "get", "--force", RESULTS_VOL_NAME, run_tag, str(dest.parent),
    ])
    if rc != 0:
        print(f"WARNING: modal volume get exit {rc}; fetch manually with "
              f"`modal volume get {RESULTS_VOL_NAME} {run_tag} {dest.parent}`")
    print(f"\nrun_tag={run_tag}\nresults: {dest}")
