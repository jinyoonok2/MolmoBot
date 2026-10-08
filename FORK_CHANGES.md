# MolmoBot fork changes and reproduction

The `rby1-compatible` branch contains the RB-Y1 model integration and evaluation fixes used with our patched MolmoSpaces fork. Its implementation is pinned at `9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb`. The original comparison baseline is `33c0ca77bf6062a23d60ffd4a6859334c4a46d30`, preserved on this fork's `main`; it is a historical upstream snapshot, not a claim about upstream main today.

The former `rby1-custom` and `rby1-long-horizon-vla-training` branches both pointed to `9c2ebfa` and were consolidated into `rby1-compatible`. That consolidation preserved code and history. Our custom dataset, training preparation, and composite navigation-to-door evaluation live in the parent `LRL_project`, rather than adding changes to MolmoBot's model or training implementation.

## Commit history from the original baseline

| Commit | Retained outcome or historical role |
|---|---|
| `40df752` | Prefer cached Hugging Face checkpoints for RB-Y1 SPOC; use an IPv4 fallback when downloading; adjust its policy constructor. This is the separate SPOC evaluation path. |
| `25e41bf` | Add multitask evaluation integration, optional-config handling, diagnostics, single-episode selection, and launcher compatibility with the installed MolmoSpaces evaluation signature. |
| `8729ddc` | Introduce the terminate-on-success evaluation override. |
| `a8c4f44` | Temporarily revert that override. |
| `6265178` | Restore the override; the final launcher exposes `--terminate_upon_success`. |
| `20d2296` | Add seeded action generation and recorded-trajectory replay for controlled evaluation. |
| `b84a58f` | Remove experimental RB-Y1 binary/inverted gripper remapping and restore the model's default gripper outputs. |
| `9c2ebfa` | Pass `task_type` to the MolmoSpaces policy base constructor, matching the selected compatible API. |

All commits remain reachable from `rby1-compatible`. Inspect them with `git show COMMIT` or `git log --reverse --oneline 33c0ca7..9c2ebfa`.

## Files changed and compatibility fixes

The retained implementation changes four files relative to the baseline, with 334 inserted and 35 deleted lines, excluding this record and its README link.

| File | Change and reason |
|---|---|
| `MolmoBot/launch_scripts/run_eval.py` | Filter keyword arguments using the installed `run_evaluation` signature; resolve `module:ClassName` configs; add `--idx`; use the config's horizon when `--task_horizon` is omitted; expose early termination on success. |
| `MolmoBot/olmo/eval/configure_molmo_spaces.py` | Pass task type to the policy base; tolerate absent optional fields; limit relative or absolute arm motion for RB-Y1 arm groups; add multitask integration and observation/action diagnostics; add seeded action generation and H5 recorded-action replay. |
| `MolmoBot-SPOC/eval/config/rby1_eval_config.py` | Resolve cached checkpoint snapshots before network access and add an IPv4 download fallback. |
| `MolmoBot-SPOC/eval/spoc_policy.py` | Adjust the SPOC constructor call for its bring-up compatibility path. The project's LLM/action-expert experiment uses the multitask policy rather than SPOC. |

The final constructor fix is [9c2ebfa](https://github.com/jinyoonok2/MolmoBot/commit/9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb): `SynthVLAPolicy` calls `super().__init__(config, task_type)`. Pair it with the patched MolmoSpaces source, not an arbitrary newer API. The robot-body/contact-scoring and legacy benchmark fixes are in [MolmoSpaces' change record](https://github.com/jinyoonok2/molmospaces/blob/rby1-compatible/FORK_CHANGES.md), especially commit `9b5f4e1`; they are not MolmoBot model-weight changes.

The final multitask policy retains default gripper handling. Earlier binary, inverted, and threshold-mapping experiments are not required patches. `RBY1_MODEL_ACTION_SEED` or `action_generator_seed` initializes a dedicated generator, resets it with the policy, and supplies it to action generation. Recorded replay is a control that plays H5 commands; it is separate from learned inference.

## Reproduce the compatible source pair

Use both forks together in a fresh directory. The immutable commits below reproduce the implementation used by this project; later commits on `rby1-compatible` add change records and README links. Existing compatible checkouts already contain the fixes.

```bash
mkdir rby1-reproduction
cd rby1-reproduction

git clone --branch rby1-compatible https://github.com/jinyoonok2/MolmoBot.git
git clone --branch rby1-compatible https://github.com/jinyoonok2/molmospaces.git

git -C MolmoBot checkout --detach 9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb
git -C molmospaces checkout --detach fabf2af3839ca59ac35913271379fb97cd60914e
```

Activate a separate Python 3.11 environment. The observed working environment used Python 3.11.15, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, MuJoCo 3.4.0, transformers 4.57.1, and pydantic 2.13.4. This is a version snapshot, not a complete dependency lock or a validated clean installation. Select CUDA packages appropriate to the host before installing the projects.

For package wiring, run from the directory containing both clones:

```bash
python -m pip install -e './molmospaces[mujoco]'
python -m pip install -e './MolmoBot/MolmoBot[eval,train]'
python -m pip install -e './molmospaces[mujoco]'
python -m pip check
```

MolmoBot's `eval` extra declares the original upstream MolmoSpaces commit `cd23bec` as a Git dependency. Installing it can replace the patched package; the final editable MolmoSpaces install restores the compatible fork. Diagnose dependency errors from `pip check` before running inference. Planner-based generation additionally requires cuRobo: the project's installed source was `87e857d46fa5398f268c7f31d26566351be8671d`, while MolmoSpaces' optional `curobo` extra declares `417c995647fcb173a2bc094d1284b2a4f4b000ad`. Preserve that distinction when rebuilding the planner environment.

Verify package locations without loading model checkpoints:

```bash
export RBY1_SOURCE_ROOT="$PWD"
export PYTHONPATH="$RBY1_SOURCE_ROOT/MolmoBot/MolmoBot:$RBY1_SOURCE_ROOT/molmospaces:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=0
export JAX_PLATFORMS=cpu
export RBY1_MODEL_ACTION_SEED=42
python -c 'import sys, olmo, molmo_spaces; print(sys.executable); print(olmo.__file__); print(molmo_spaces.__file__)'
```

`olmo` must resolve inside the MolmoBot clone and `molmo_spaces` inside the MolmoSpaces clone. Seed 42 controls model action sampling, not all physics or rendering variation. Use a GPU allocation for rendering/inference and make the released checkpoint, scene assets, benchmark JSON, object assets, and grasp resources available before evaluation.

## Verify a benchmark episode

The historical RB-Y1 benchmark-loading setup used `molmospaces-bench-v2/20260327`. Verify the actual resource directories before using it: benchmark versions and assets are separate from Git source revisions. Required task directories are:

| Task | Directory within the benchmark release |
|---|---|
| Pick | `procthor-objaverse/rby1_benchmarks/pick_benchmark` |
| Pick and place | `procthor-objaverse/rby1_benchmarks/pnp_benchmark` |
| Opening | `ithor/rby1_benchmarks/opening_benchmark` |
| Door opening | `procthor-10k/rby1_benchmarks/door_opening_benchmark` |

The following is a single door-opening benchmark example, run from the directory containing both clones. Replace the paths with existing checkpoint, benchmark, resource-cache, asset, and output locations. It evaluates the released policy on a standard benchmark; it does not evaluate the parent's custom navigation-to-door task.

```bash
export RBY1_CHECKPOINT_DIR=/absolute/path/to/MolmoBot-RBY1Multitask
export RBY1_BENCHMARK_DIR=/absolute/path/to/door_opening_benchmark
export RBY1_EVAL_OUTPUT=/absolute/path/to/new/evaluation-output
export MLSPACES_CACHE_DIR=/absolute/path/to/molmo-spaces-resources
export MLSPACES_ASSETS_DIR=/absolute/path/to/molmospaces/assets

python MolmoBot/MolmoBot/launch_scripts/run_eval.py \
  --checkpoint_path "$RBY1_CHECKPOINT_DIR" \
  --benchmark_path "$RBY1_BENCHMARK_DIR" \
  --eval_config_cls olmo.eval.configure_molmo_spaces:MolmoBotRBY1DoorEvalConfig \
  --idx 0 --task_horizon 400 --terminate_upon_success \
  --output_dir "$RBY1_EVAL_OUTPUT"
```

To preview required assets for that episode before installing them, run:

```bash
python molmospaces/scripts/benchmarks/prepare_benchmark_assets.py \
  --benchmark_dir "$RBY1_BENCHMARK_DIR" --idx 0 --dry_run
```

Remove `--dry_run` to prepare missing assets in the configured storage location. The helper loads benchmark metadata; a successful preview does not verify model inference.

Check episode success, recorded H5 trajectories, and videos. A clean process exit verifies execution, not task success or reproduction of published success rates. Training and evaluation on our navigation-to-door dataset additionally use the parent `LRL_project` extensions, configs, and launchers. See [the project setup guide](https://github.com/jinyoonok2/LRL_project/blob/rby1-long-horizon-vla-training/docs/SETUP.md) for storage, asset configuration, Slurm, and the separate extension boundary.

## Rebuild the retained patch on the original baseline

For an independent source reconstruction, run these commands from the directory containing the clones. Use an unused worktree directory. The patch is the net retained change, so reverted experiments are automatically excluded.

```bash
git -C MolmoBot diff --binary \
  33c0ca77bf6062a23d60ffd4a6859334c4a46d30 \
  9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb > molmobot-rby1.patch

git -C MolmoBot worktree add --detach ../MolmoBot-rebuilt \
  33c0ca77bf6062a23d60ffd4a6859334c4a46d30
git -C MolmoBot-rebuilt apply --check ../molmobot-rby1.patch
git -C MolmoBot-rebuilt apply ../molmobot-rby1.patch
git -C MolmoBot-rebuilt add -A
git -C MolmoBot-rebuilt diff --cached --exit-code \
  9c2ebfa5cafcf13f9664e70703e25de38e1d5fcb
```

The last command must produce no diff and exit with status 0. It checks that the reconstructed tracked files equal the compatible implementation. The worktree has the original commit checked out plus staged patches; it does not recreate the historical commit IDs. To preserve the original history, use the fork and exact target commit directly.
