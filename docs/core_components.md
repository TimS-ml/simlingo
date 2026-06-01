# SimLingo — Core Components Reference

Per-module reference for the **full SimLingo** (`simlingo_training/`) plus the CARLA
inference stack (`team_code/`). For the big picture see `architecture_overview.md`; for
the closed-loop trace see `model_inference_flow.md`. File:line references point at the
current tree.

---

## 1. Vision encoder

**Files:** `simlingo_training/models/encoder/vlm.py`,
`simlingo_training/models/encoder/internvl2_model.py`,
`simlingo_training/utils/internvl2_utils.py`

Wraps the pretrained `OpenGVLab/InternVL2-1B`. Keeps the **InternViT** vision tower and
its `mlp1` projector; the bundled language model is discarded (`vlm.py:30-31`) because
SimLingo plugs in its own `LLM` wrapper.

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `VLMEncoderModel` | `vlm.py:6` | Top-level encoder wrapper; freeze logic (`:36-44`) freezes all params except `mlp1` |
| `LingoInternVLModel` | `internvl2_model.py:6` | Loads `AutoModel(...InternVL2..., trust_remote_code=True)`; records vocab size |
| `replace_placeholder_tokens` | `internvl2_model.py:17` | **Core**: (a) MLP-encode waypoint placeholder tokens via `wp_encoder`; (b) run InternViT, overwrite `<IMG_CONTEXT>` slots with vision embeddings |
| `get_num_image_tokens_per_patch` | `internvl2_utils.py:21` | `(image_size//patch)² · downsample²` → **256 tokens/tile** for InternVL2-1B |
| `preprocess_image_batch` | `internvl2_utils.py:179` | Tile (`dynamic_preprocess`) + ImageNet-normalize a batch |
| `dynamic_preprocess` | `internvl2_utils.py:231` | InternVL aspect-ratio tiling into 448×448 crops |
| `get_custom_chat_template` | `internvl2_utils.py:94` | Build InternLM2 chat prompt; expand `<image>` → `<img>` + `<IMG_CONTEXT>·N` + `</img>` |

**I/O:** front image `[B,1,N=2,C,448,448]` → 512 vision embeddings spliced into the LLM
sequence. The splice uses `inputs_embeds[selected] = inputs_embeds[selected]*0.0 +
vit_embeds` (`internvl2_model.py:124`) — zero-then-add keeps it differentiable so
gradients reach `mlp1`.

> The front camera is split into a **1×2 grid (2 tiles)**, 256 tokens each = **512 image
> tokens** (`datamodule.py:110`, `NUM_IMAGE_PATCHES=2`). `T==1` only — no multi-frame.

---

## 2. Language model head

**File:** `simlingo_training/models/language_model/llm.py`

Thin wrapper that extracts the **InternLM2** decoder from InternVL2 and drives it with
`inputs_embeds`.

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `LLM.__init__` | `llm.py:50` | Loads `AutoModel(...).language_model`; rebinds `embed_tokens` (`:90-93`); optional LoRA (`:106-119`); caches `hidden_size`, `vocab_size` |
| `LLM.forward` | `llm.py:126` | Calls inner model with `inputs_embeds=...`, `output_hidden_states=True`; returns `(features=hidden_states[-1], logits)` |
| `LLM.sample_categorical` | `llm.py:145` | Temperature / top-k / top-p sampling (`<=0` ⇒ greedy argmax) |
| `LLM.greedy_sample` | `llm.py:178` | Autoregressive generation **in embedding space**: forward → project last hidden via external `logit_matrix` → sample → re-embed via external `input_embed_matrix` → append; stops at EOS |

> `greedy_sample` needs `input_embed_matrix` / `logit_matrix` passed in — the caller
> supplies `LanguageAdaptor.embed_tokens.weight` and `.lm_head.weight`.

---

## 3. Adaptors (modality bridges)

**File:** `simlingo_training/models/adaptors/adaptors.py`

Convert heterogeneous inputs into embedding tokens and decode LLM hidden states back to
predictions; own the per-task losses.

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `WaypointInputAdaptor` | `adaptors.py:64` | MLP `[B,N,2] → [B,N,H]`; **encodes** coordinates into LLM token space (used as `wp_encoder`) |
| `DrivingAdaptor` | `adaptors.py:96` | Learnable query embeds + regression heads. `forward` (`:139`) builds query tokens; `get_predictions` (`:163`) slices hidden states → heads → `.cumsum(1)`; `compute_loss` (`:183`) smooth-L1 |
| `LanguageAdaptor` | `adaptors.py:224` | Captures `embed_tokens` + resolves `lm_head` (`:232`); `forward` embeds `phrase_ids`; `compute_loss` (`:259`) shifted next-token cross-entropy (`ignore_index=-1`) |
| `AdaptorList` | `adaptors.py:276` | Concatenates language+driving tokens, applies a stable permutation pushing invalid tokens to the end (`:322-330`); `split_outputs_by_adaptor` (`:357`) inverts it |

**Head dims:** `route_head` → `[B,20,2]` (`future_waypoints=20`); `speed_wps_head` →
`[B,10,2]` with `speed_wps_mode='2d'` (`future_speed_waypoints=10`). Both predict
per-step deltas → `cumsum` → absolute trajectory.

> Latent bug: with `predict_route_as_wps=False` the code would `AttributeError` on
> `self.queries['speed_wps']` (`:134`) because those structures are only built inside the
> `predict_route_as_wps` block. The config default is `True`, so this never fires.

---

## 4. DrivingModel (the assembly)

**Files:** `simlingo_training/models/driving.py` (732 lines),
`simlingo_training/models/utils.py`

`DrivingModel(pl.LightningModule)` (`driving.py:40`) is the center. `__init__`
(`:41-101`) Hydra-instantiates the vision model and LLM, builds the `DrivingAdaptor` /
`AdaptorList` / `WaypointInputAdaptor`.

| Method | `file:line` | Role |
|---|---|---|
| `forward` | `driving.py:104` | **Inference sampler** — per-item greedy language sampling, then a second pass to regress the trajectory; returns `(speed_wps, route, language)` |
| `forward_model` | `driving.py:190` | **Shared training pass** — `replace_placeholder_tokens` → LLM → split off the trailing adaptor block |
| `forward_loss` | `driving.py:236` | Runs `forward_model` then `AdaptorList.compute_loss`; `summarise_losses` → `TrainingOutput` |
| `training_step` / `validation_step` | `driving.py:263` / `:274` | Standard Lightning steps with logging |
| `predict_step` | `driving.py:285` | Eval — runs `forward`, resamples route via `equal_spacing_route`, accumulates predictions |
| `on_predict_epoch_end` | `driving.py:344` | Dumps language predictions + computes the **dreamer success rate** per mode (`:486-705`) |
| `configure_optimizers` | `driving.py:718` | `AdamW(lr=5e-2, wd=0.1)` + `OneCycleLR` (per-step) |
| `summarise_losses` | `utils.py:7` | Per-key `sum/count` average, total = unweighted sum of averages |

> Multi-task losses are an **unweighted sum of per-task averages** — there is no explicit
> loss weighting (`weights=None`).

---

## 5. Data pipeline

**Files:** `simlingo_training/dataloader/dataset_base.py` (919 lines),
`dataset_driving.py`, `datamodule.py`

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `BaseDataset` | `dataset_base.py:32` | Indexing & raw loaders shared by all task datasets; walks `database/`, filters routes by infraction score (`:255`), builds flat per-frame index arrays |
| `BaseDataset.get_waypoints` | `dataset_base.py:785` | Transform future ego matrices into current ego BEV frame (x forward, y lateral) |
| `BaseDataset.get_navigational_conditioning` | `dataset_base.py:484` | Build `target_options` (target-point text / `<TARGET_POINT>` placeholder / "Command: …") |
| `Data_Driving.__getitem__` | `dataset_driving.py:33` | Assemble one driving sample → `DatasetOutput`; builds prompt/answer (`:240-261`) |
| `DataModule` | `datamodule.py:60` | LightningDataModule; instantiates per-bucket datasets, `WeightedRandomSampler`, holds processor/tokenizer (adds `<WAYPOINTS>`/`<ROUTE>`/`<TARGET_POINT>`) |
| `DataModule.dl_collate_fn` | `datamodule.py:310` | **Collate**: image tiling, chat-template tokenization (assistant-only loss mask), batch into `DrivingExample` |

**Key shapes:** `pred_len=11`, `hist_len=1` → `waypoints [B,10,2]`; route/path `[B,20,2]`;
camera `[B,1,2,3,448,448]`. Bucket balancing oversamples rare maneuvers
(acceleration/lateral/recovery); prompt-type probabilities are renormalized at runtime.

> Coordinate frame: waypoints/route are in the **current ego BEV frame**; augmentation
> (yaw + lateral shift) is applied consistently to waypoints, route, target points, and
> boxes, matching the shifted camera image.

---

## 6. Dreaming & language datasets

**Files:** `simlingo_training/dataloader/dataset_dreamer.py`,
`dataset_eval_qa_comm.py`, `dataset_eval_dreamer.py`

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `Data_Dreamer` | `dataset_dreamer.py:21` | Action-Dreaming training samples: same image + sampled instruction + one of several alternative trajectories + safety flag |
| `Eval_Dreamer` | `dataset_eval_dreamer.py:21` | Dreaming eval twin; additionally emits `eval_infos` (mode, allowed, org vs new trajectories) |
| `Data_Eval` | `dataset_eval_qa_comm.py:16` | VQA / commentary eval samples from fixed `evalset_*.json` |

**Action Dreaming mechanism** (`dataset_dreamer.py:88-162`): open the per-frame
alternatives file, `random.choice` an option (instruction + waypoints + `safe_to_execute`),
resolve the `'org'` sentinel to the original safe trajectory, and prefix the prompt with
`<SAFETY>` or `<INSTRUCTION_FOLLOWING>`. Under `<SAFETY>` an unsafe instruction is
**refused** — supervised target reverts to the original trajectory and the answer becomes
the refusal text. This is the core language-action-alignment signal.

---

## 7. Training, config & visualization

**Files:** `simlingo_training/train.py`, `config.py`, `config/`,
`callbacks/visualise.py`

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `main` | `train.py:47` | `@hydra.main` entry — processor, DataModule, DrivingModel, Trainer, `trainer.fit` |
| Trainer setup | `train.py:160-217` | DeepSpeed ZeRO-2, `precision='16-mixed'`, `gradient_clip_val=0.3`, wandb |
| `register_configs` | `config.py:306` | Registers dataclass schemas into Hydra `ConfigStore` |
| `TrainConfig` / `DrivingModelConfig` / `DatasetBaseConfig` | `config.py:247` / `:75` / `:108` | Structured config defaults |
| `VisualiseCallback` | `visualise.py:141` | Periodic sampled forward → GT-vs-pred waypoint/route plots + language to wandb |

**Defaults / gotchas:** schema `lr=5e-2` is overridden to `3e-5` by every experiment yaml;
training is **epoch-driven** (`max_epochs`, `max_steps` commented out); `_recursive_=False`
on both `instantiate` calls so modules build their own children; `gradient_clip_val` and
the callback interval are hard-coded in `train.py`, not config-driven.

---

## 8. Language evaluation

**Files:** `simlingo_training/eval.py`, `eval_metrics.py`,
`simlingo_training/utils/gpt_eval.py`

| Symbol | `file:line` | Responsibility |
|---|---|---|
| `eval.py::main` | `eval.py:38` | Pick `eval_mode ∈ {QA, commentary, Dreaming}` (`:64-66`), load checkpoint, disable augmentation, `trainer.predict` |
| `evaluation_suit` | `eval_metrics.py:37` | Accuracy + NLG (BLEU/ROUGE-L/CIDEr/METEOR/SPICE) + GPT-4 judge |
| `gpt_forward` | `gpt_eval.py:23` | GPT-4-as-judge worker: "Rate my answer out of 100" vs GT |

**Closed-loop-free**: open-loop predictions vs logged GT + rule-based dreamer success.

> Gotchas: eval mode is selected by **uncommenting** lines (`eval.py:64-66`); the GPT key
> is a hard-coded placeholder (`gpt_eval.py:5`); exact-match accuracy is strict string
> equality. The dreamer success loop only scores the `<SAFETY>` subset due to a `zip`
> length mismatch (`driving.py:486`).

---

## 9. Closed-loop inference agent

**File:** `team_code/agent_simlingo.py` (1170 lines)

`LingoAgent(AutonomousAgent)` (`:83`) runs in the CARLA Leaderboard on `Track.SENSORS`.

| Method | `file:line` | Role |
|---|---|---|
| `setup` | `:106` | Load Hydra config, build processor + tokens, Hydra-instantiate `DrivingModel` (bf16), `load_state_dict`, build PIDs + UKF |
| `sensors` | `:354` | RGB camera(s) + IMU + GNSS + speedometer |
| `tick` | `:426` | Image preprocessing + UKF localization + route/target point + prompt + tokenization → `DrivingInput` |
| `run_step` | `:762` | tick → `self.model(...)` → `control_pid` → safety/creep → `carla.VehicleControl` |
| `control_pid` | `:915` | Waypoints → (steer, throttle, brake) |
| UKF helpers | `:1024-1158` | `bicycle_model_forward` (fx), measurement/residual/mean functions |

> No SLAM: localization is GPS/IMU/speed fused by an **Unscented Kalman Filter**; the
> route is the privileged `GlobalRoutePlanner` `_global_plan`. The model is conditioned on
> the next ego-frame **target points** via the `<TARGET_POINT>` placeholder.

---

## 10. Control & planning

**Files:** `team_code/nav_planner.py`, `lateral_controller.py`,
`longitudinal_controller.py`, `kinematic_bicycle_model.py`, `privileged_route_planner.py`

| Symbol | `file:line` | Role |
|---|---|---|
| `RoutePlanner` | `nav_planner.py:180` | GPS→CARLA (Mercator), pops passed waypoints, returns upcoming `(pos, RoadOption)` |
| `LateralPIDController` (used) | `nav_planner.py:73` | Speed-adaptive lookahead + heading-error PID → steer; gains `k_p=3.118, k_d=1.378, k_i=0.641` |
| `LongitudinalLinearRegressionController` | `longitudinal_controller.py:184` | Expert throttle/brake model (data collection) |
| `KinematicBicycleModel` | `kinematic_bicycle_model.py:83` | Expert collision forecasting; the agent re-implements the same kinematics inline for its UKF |
| `PrivilegedRoutePlanner` | `privileged_route_planner.py:38` | Expert navigation during data collection (supersampled route, TL/stop distances, leading-vehicle detection) |

> Important separation: the **deployed agent** uses `nav_planner.LateralPIDController` +
> `transfuser_utils.PIDController` for speed. The `lateral_controller.py` /
> `longitudinal_controller.py` / `kinematic_bicycle_model.py` / `privileged_route_planner.py`
> classes are the **PDM-Lite expert stack** used for *data collection*, not VLA inference.

---

## 11. SimLingo-Base (CarLLaVA)

**Folder:** `simlingo_base_training/`

Vision-only baseline. Same DETR-style query-token architecture, but:

- **Encoder:** `LLaVAnextEncoderModel` (`encoder/llavanext.py:30`) or `ResnetEncoderModel`
  (`encoder/resnet.py:28`, `microsoft/resnet-34`).
- **LLM:** `Llama` built **from scratch** from a `CONFIGS` size dict (default `x-small`
  235M); `embed_tokens=None` (`language_model/llama.py:88`).
- **Adaptors:** `DrivingAdaptor` only, plus a `VectorInputAdaptor` (`adaptors.py:75`) that
  encodes scalar speed as a token; **MSE** loss (`adaptors.py:225`).
- **Optimizer:** `FusedAdam` with two regex param groups (separate vision LR),
  `DrivingModel.configure_optimizers` (`driving.py:382`).
- **Output:** `(speed_wps, route)` — no language generation.

See the contrast table in `architecture_overview.md` §1.
