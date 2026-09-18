# Workspace, ownership and verified commands

## Workspace

Original checkout `/Users/wesleylu/Projects/Research/structure_descent` on branch `goleu-redesign`
was **not modified**: no commit, no stash, no reset, no clean, nothing pushed.  Everything below lives
in sibling worktrees created from `6a1774e`.

| worktree | branch | head | owner |
|---|---|---|---|
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/00-foundation` | `exp/consequence-20260910T094713Z-31092/00-foundation` | `b6307f0` | coordinator |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/01-grounded` | `exp/consequence-20260910T094713Z-31092/01-grounded` | `2ac6823` | grounded |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/02-axis` | `exp/consequence-20260910T094713Z-31092/02-axis` | `a2b41ac` | axis |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/03-interventions` | `exp/consequence-20260910T094713Z-31092/03-interventions` | `72463bc` | interventions |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/04-complementarity` | `exp/consequence-20260910T094713Z-31092/04-complementarity` | `8e50014` | complementarity |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/05-uncertainty` | `exp/consequence-20260910T094713Z-31092/05-uncertainty` | `e730bc0` | uncertainty |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/06-transfer` | `exp/consequence-20260910T094713Z-31092/06-transfer` | `779ac0a` | transfer |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/07-gate` | `exp/consequence-20260910T094713Z-31092/07-gate` | `a760811` | gate |
| `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/90-integration` | `exp/consequence-20260910T094713Z-31092/90-integration` | `91cc3b4` | coordinator |

Manifest: `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts/workspace_manifest.json`.  Artifacts, per-event predictions and the status ledger:
`/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts/`.  Nothing was imported from the original working tree: it was clean when the worktrees were
created, so the foundation starts from committed content only.

## Environment

Python `/Users/wesleylu/Projects/Research/structure_descent/venv/bin/python` is shared by every
worktree, which keeps dependency versions identical across variants; each worktree has its own
`logs/`, `tmp/` and source tree, and `PYTHONPATH` selects the worktree's own package.  This is a
deviation from a separate virtual environment per worktree, taken because the machine has 86 GB free
and eight torch installations would not fit; the dependency fingerprint is recorded in every artifact.
Raw data, records and generation caches are read from the original checkout and never written to.

## Commands

```bash
# audit (dataset semantics, ordering support, availability)
cd /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-worktrees/90-integration && PYTHONPATH=. venv=/Users/wesleylu/Projects/Research/structure_descent/venv/bin/python && \
  PYTHONPATH=. $venv -m omleu_experiments.cli audit --datasets optima lpmc swissmetro

# prepare deterministic sentence sources (templates, identity text); no generator is called
PYTHONPATH=. $venv -m omleu_experiments.cli prepare --datasets optima lpmc swissmetro --seeds 7 11 13 \
  --sources template identity --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# the core matrix under the primary protocol (the run performed in this session)
PYTHONPATH=. $venv -m omleu_experiments.cli run --suite core_matrix \
  --datasets lpmc optima swissmetro --seeds 7 11 13 --protocols person_disjoint \
  --folds 3 --members 5 --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# extended matrix (adds the axis ablations, mixture-aware training and the combined models)
PYTHONPATH=. $venv -m omleu_experiments.cli run --suite core_matrix_extended --datasets lpmc \
  --seeds 7 --protocols person_disjoint --folds 3 --members 5 --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# secondary protocol: supported on LPMC only
PYTHONPATH=. $venv -m omleu_experiments.cli run --suite core_matrix --datasets lpmc --seeds 7 \
  --protocols temporal --folds 3 --members 5 --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# learning curves and the six ordered transfers
PYTHONPATH=. $venv -m omleu_experiments.cli run --suite transfer --datasets lpmc optima swissmetro \
  --seeds 7 --protocols person_disjoint --folds 3 --members 5 --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# optional conditional gate
PYTHONPATH=. $venv -m omleu_experiments.cli run --suite optional --datasets lpmc --seeds 7 \
  --protocols person_disjoint --folds 3 --members 5 --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# tables and reports, recomputed from saved predictions
PYTHONPATH=. $venv -m omleu_experiments.cli evaluate --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts
PYTHONPATH=. $venv -m omleu_experiments.cli report   --artifact-root /Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts

# every acceptance test
PYTHONPATH=. $venv -m pytest omleu_experiments/tests -q
```

## Blocked experiments and what unblocks them

| experiment | status | to start it |
|---|---|---|
| E1 grounded generation | `blocked_generation_budget` | `export OMLEU_GENERATION_BUDGET='{"requests": 6000, "provider": "ollama", "model": "qwen3-vl:8b-instruct"}'`; the manifest at `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts/e1_generation/generation_manifest.json` gives the request counts |
| E1 human audit | `blocked_annotations` | annotate `/Users/wesleylu/Projects/Research/structure_descent-consequence-20260910T094713Z-31092-artifacts/e1_audit/audit_export_<dataset>_seed7.json`; the file is label-blind and has the four required fields |
| E3 training runs | `not_run` | merge the axis reader with the consistency losses and add a matrix slot; the loss interface is tested |
| E5 empirical | `blocked_data` | a per-alternative outcome distribution, repeated journeys, or elicited beliefs |
| E6, E7 suites | `not_run` | the two commands above; a single training process was reserved for the core matrix |
