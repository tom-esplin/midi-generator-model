"""Unified training orchestration for all model families.

Exposes a single `train_model` loop that handles GRU and Transformer variants
via a lightweight config/adapter scheme, plus a `run_all_training` driver that
iterates over genres, dataset modes, and model configs and collects losses
into a nested dictionary.
"""

import gc
import json
import os
import shutil
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from miditok import PerTok

from training.prep_dataset import MidiDataset, ContinuousMidiDataset
from models.gru import OptimizedGru, GRUModel
from models.transformer import OptimizedTransformer, TransformerDecoder

try:
    from google.colab import files as colab_files
    _HAS_COLAB = True
except ImportError:
    colab_files = None
    _HAS_COLAB = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")
print(torch.version.cuda)
print(torch.__version__)


CHUNK_SIZE = 1000
CHECKPOINT_DIR = Path("models", "model_weights")
CHECKPOINT_INTERVAL_SECONDS = 360000
LOSS_MIRROR_DIRS_DEFAULT = [
    Path("models", "model_weights", "losses"),
    Path("/content/drive/MyDrive/midi-generator-model/losses"),
    Path(tempfile.gettempdir()) / "midi_generator_losses",
]
CHECKPOINT_MIRROR_DIRS_DEFAULT = [
    Path("models", "model_weights"),
    Path("/content/drive/MyDrive/midi-generator-model/model_weights"),
]


def _flush_filesystem():
    """Ask the OS to flush buffered writes to underlying storage.

    On Colab this forces the Drive FUSE driver to push anything it's still
    buffering up to Google, which is critical if the VM then dies unexpectedly
    (power cut, preemption, disconnect).
    """
    try:
        os.sync()
    except (AttributeError, OSError):
        pass


def _resolve_mirror_dirs(candidate_dirs, label):
    """Probe each candidate dir for writability, returning the usable subset."""
    resolved = []
    for d in candidate_dirs:
        d = Path(d)
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            resolved.append(d)
        except Exception as exc:  # noqa: BLE001
            print(f"[{label}] skipping unreachable mirror {d}: {exc}")
    return resolved


def free_memory(verbose=False):
    """Release Python garbage, empty the CUDA allocator cache, and report if asked."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if verbose:
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved = torch.cuda.memory_reserved() / 1024 ** 3
            peak = torch.cuda.max_memory_allocated() / 1024 ** 3
            print(f"[cuda] allocated={allocated:.2f} GiB | reserved={reserved:.2f} GiB | peak={peak:.2f} GiB")


def _reset_memory_stats():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Generic sampling utility (works across all model configs)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model,
    start_tokens,
    max_len,
    temperature=1.0,
    top_k=None,
    needs_mask=False,
    needs_hidden=False,
    hidden_shape=None,
):
    """Generate tokens autoregressively from any model in MODEL_CONFIGS.

    Args:
        model: the module to sample from.
        start_tokens: Long tensor of shape (batch, seq_len) to seed generation.
        max_len: number of new tokens to generate.
        temperature: softmax temperature.
        top_k: optional top-k filtering.
        needs_mask: if True, build a causal mask each step.
        needs_hidden: if True, maintain a hidden state for an RNN-style model.
        hidden_shape: (num_layers, hidden_dim) used when `needs_hidden` is True.
    """
    model.eval()
    model.to(device)
    context = start_tokens.to(device)

    hidden = None
    if needs_hidden and hidden_shape is not None:
        num_layers, hidden_dim = hidden_shape
        hidden = torch.zeros(num_layers, context.size(0), hidden_dim, device=device)

    for _ in range(max_len):
        if needs_mask:
            mask = nn.Transformer.generate_square_subsequent_mask(context.size(1)).to(device)
            logits = model(context, mask=mask)
        elif needs_hidden:
            logits, hidden = model(context[:, -1:], hidden)
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)
        else:
            logits = model(context)

        next_token_logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
            next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        context = torch.cat((context, next_token), dim=1)

    return context


# ---------------------------------------------------------------------------
# Internal per-step helpers (shared by the unified training loop)
# ---------------------------------------------------------------------------

def _full_sequence_step(model, x, y, cfg, vocab_size, causal_mask, hidden,
                        optimizer, scaler, loss_fn):
    """One optimizer step when the model consumes the full sequence at once."""
    use_amp = scaler is not None
    optimizer.zero_grad(set_to_none=True)

    with autocast(device.type, dtype=torch.float16, enabled=use_amp):
        if cfg["needs_mask"]:
            logits = model(x, mask=causal_mask)
        elif cfg["needs_hidden"]:
            logits, _ = model(x, hidden)
        else:
            logits = model(x)
        loss = loss_fn(logits.reshape(-1, vocab_size), y.flatten())

    if use_amp:
        scaler.scale(loss).backward()
        if cfg.get("grad_clip"):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if cfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        optimizer.step()

    loss_val = loss.item()
    del logits, loss
    return loss_val


def _stepwise_step(model, x, y, cfg, optimizer, loss_fn):
    """One outer step when the model must be trained token-by-token (custom GRU)."""
    hidden = torch.zeros(cfg["num_layers"], x.size(0), cfg["hidden_dim"], device=device)
    seq_len = x.size(1)
    token_losses = []

    for t in range(seq_len):
        optimizer.zero_grad(set_to_none=True)
        logits, hidden = model(x[:, t:t + 1], hidden)
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        loss = loss_fn(logits, y[:, t])
        loss.backward()
        optimizer.step()
        hidden = hidden.detach()
        token_losses.append(loss.item())
        del logits, loss

    del hidden
    return sum(token_losses) / max(1, len(token_losses))


# ---------------------------------------------------------------------------
# Unified training loop
# ---------------------------------------------------------------------------

def train_model(model, dataloader, cfg, vocab_size, n_steps, eval_interval=1000,
                checkpoint_seconds=CHECKPOINT_INTERVAL_SECONDS, recorder=None,
                checkpoint_mirror_dirs=None):
    """Train `model` for `n_steps` optimizer iterations using the given config.

    The config encodes model-specific behaviour so that this single loop can
    drive OptimizedGru, GRUModel, OptimizedTransformer, and TransformerDecoder.
    """
    model.to(device).train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = cfg["optimizer_factory"](model)
    scheduler = (
        cfg["scheduler_factory"](optimizer, n_steps)
        if cfg.get("scheduler_factory") is not None
        else None
    )
    use_amp = cfg.get("use_amp", False) and device.type == "cuda"
    scaler = GradScaler(device=device.type) if use_amp else None

    losses = []
    start_time = time.time()
    data_iter = iter(dataloader)
    causal_mask_cache = {}

    pbar = tqdm(range(n_steps), desc=cfg["name"])
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = batch.to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        if cfg.get("stepwise"):
            step_loss = _stepwise_step(model, x, y, cfg, optimizer, loss_fn)
        else:
            causal_mask = None
            if cfg.get("needs_mask"):
                seq_len = x.size(1)
                causal_mask = causal_mask_cache.get(seq_len)
                if causal_mask is None:
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
                    causal_mask_cache[seq_len] = causal_mask

            hidden = None
            if cfg.get("needs_hidden"):
                hidden = torch.zeros(
                    cfg["num_layers"], batch.size(0), cfg["hidden_dim"], device=device
                )

            step_loss = _full_sequence_step(
                model, x, y, cfg, vocab_size, causal_mask, hidden,
                optimizer, scaler, loss_fn,
            )

        losses.append(step_loss)
        pbar.set_postfix({"loss": f"{step_loss:.4f}"})

        if recorder is not None:
            recorder.record_step(step, step_loss)

        if scheduler is not None:
            scheduler.step()

        if (step + 1) % eval_interval == 0:
            print(f"\n[{cfg['name']}] Step {step + 1} | Loss {step_loss:.4f}")

        if time.time() - start_time >= checkpoint_seconds:
            _save_intermediate_checkpoint(
                model, optimizer, cfg["name"], step,
                checkpoint_mirror_dirs or [CHECKPOINT_DIR],
            )
            start_time = time.time()

        del batch, x, y

    del optimizer, scheduler, scaler, data_iter, causal_mask_cache, loss_fn
    free_memory()
    return losses


# ---------------------------------------------------------------------------
# Model configuration registry
# ---------------------------------------------------------------------------

def build_model_configs(vocab_size, chunk_size=CHUNK_SIZE):
    """Return the list of training configs used for every (genre, dataset) combo."""
    num_layers = 5
    hidden_dim = 512
    embedding_dim = 512
    d_model = 512
    d_ff = 2048
    nhead = 16
    lr_transformer = 3e-4

    return [
        {
            "name": "optimized_gru",
            "factory": lambda: OptimizedGru(vocab_size, embedding_dim, hidden_dim, num_layers),
            "optimizer_factory": lambda m: Adam(m.parameters(), lr=1e-3),
            "scheduler_factory": None,
            "needs_mask": False,
            "needs_hidden": True,
            "stepwise": False,
            "use_amp": False,
            "grad_clip": None,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        },
        {
            "name": "homebrew_gru",
            "factory": lambda: GRUModel(vocab_size, embedding_dim, hidden_dim, num_layers),
            "optimizer_factory": lambda m: Adam(m.parameters(), lr=1e-3),
            "scheduler_factory": None,
            "needs_mask": False,
            "needs_hidden": True,
            "stepwise": True,
            "use_amp": False,
            "grad_clip": None,
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
        },
        {
            "name": "optimized_transformer",
            "factory": lambda: OptimizedTransformer(
                vocab_size, d_model=d_model, nhead=nhead,
                num_layers=num_layers, max_seq_len=chunk_size,
            ),
            "optimizer_factory": lambda m: AdamW(
                m.parameters(), lr=lr_transformer,
                weight_decay=0.01, betas=(0.9, 0.95),
            ),
            "scheduler_factory": lambda opt, total_steps: OneCycleLR(
                opt, max_lr=lr_transformer, total_steps=total_steps,
                pct_start=0.1, anneal_strategy="cos",
            ),
            "needs_mask": True,
            "needs_hidden": False,
            "stepwise": False,
            "use_amp": True,
            "grad_clip": 1.0,
            "batch_size": 16,
        },
        {
            "name": "homebrew_transformer",
            "factory": lambda: TransformerDecoder(
                vocab_size, N=num_layers, d_model=d_model, d_ff=d_ff, h=nhead,
            ),
            "optimizer_factory": lambda m: Adam(m.parameters(), lr=5e-4),
            "scheduler_factory": None,
            "needs_mask": True,
            "needs_hidden": False,
            "stepwise": False,
            "use_amp": False,
            "grad_clip": None,
            "batch_size": 8,
        },
    ]


# ---------------------------------------------------------------------------
# Checkpoint / loss persistence helpers
# ---------------------------------------------------------------------------

def _safe_name(value):
    return str(value).replace("/", "_").replace("\\", "_")


def _atomic_write_json(path, payload):
    """Write JSON to `path` atomically so a crash mid-write can't corrupt the file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_torch_save(obj, path):
    """Serialize a torch object to `path` via tmp + rename, fsync'd."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    try:
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.replace(tmp, path)


def _save_to_mirrors(obj, filename, mirror_dirs, kind="checkpoint"):
    """Save a torch state_dict (or any torch-saveable object) to every mirror.

    Writes the primary mirror atomically, then copies the finalized file to
    each additional mirror atomically. Returns list of (dir, path) tuples that
    succeeded.
    """
    if not mirror_dirs:
        return []

    primary_dir = mirror_dirs[0]
    primary_dir.mkdir(parents=True, exist_ok=True)
    primary_path = primary_dir / filename
    try:
        _atomic_torch_save(obj, primary_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[{kind}] primary save to {primary_path} failed: {exc}")
        return []

    saved = [(primary_dir, primary_path)]
    for d in mirror_dirs[1:]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / filename
            tmp_dst = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copyfile(primary_path, tmp_dst)
            try:
                with open(tmp_dst, "rb") as f:
                    os.fsync(f.fileno())
            except OSError:
                pass
            os.replace(tmp_dst, dst)
            saved.append((d, dst))
        except Exception as exc:  # noqa: BLE001
            print(f"[{kind}] mirror {d} failed: {exc}")

    _flush_filesystem()
    return saved


def _save_intermediate_checkpoint(model, optimizer, name, step, mirror_dirs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _save_to_mirrors(
        model.state_dict(),
        f"{name}_model_{ts}_{step}.pt",
        mirror_dirs,
        kind="intermediate-model",
    )
    _save_to_mirrors(
        optimizer.state_dict(),
        f"{name}_optimizer_{ts}_{step}.pt",
        mirror_dirs,
        kind="intermediate-optimizer",
    )


def _save_final_checkpoint(model, name, genre, dataset_mode, mirror_dirs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{_safe_name(dataset_mode)}_{_safe_name(genre)}_{ts}.pt"
    saved = _save_to_mirrors(
        model.state_dict(), filename, mirror_dirs, kind="final-model",
    )
    if saved:
        print(f"[final-model] saved {filename} to:")
        for _, p in saved:
            print(f"  - {p}")
    return [p for _, p in saved]


def _save_losses_snapshot(losses_dict, tag, mirror_dirs=None):
    """Write a consolidated loss-dict JSON snapshot into each mirror dir."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = _safe_name(tag)
    name = f"losses_{safe_tag}_{ts}.json"
    payload = json.dumps(losses_dict, indent=2, default=str)

    targets = list(mirror_dirs) if mirror_dirs else [CHECKPOINT_DIR]
    saved = []
    for d in targets:
        try:
            _atomic_write_json(Path(d) / name, payload)
            saved.append(Path(d) / name)
        except Exception as exc:  # noqa: BLE001
            print(f"[losses-snapshot] mirror {d} failed: {exc}")
    _flush_filesystem()
    return saved


class LossRecorder:
    """Redundant, crash-resilient recorder for training losses.

    Mirrors every loss value to one or more destinations so that:
      - Each optimizer step is appended to a per-model `.jsonl` stream and
        flushed immediately (rebuilds the loss list even if the process dies
        between consolidated snapshots).
      - The full nested state dict is atomically re-written on a schedule
        and whenever a model starts/finishes (tmp file + os.replace so a
        partial write never clobbers the previous valid snapshot).
      - Any mirror that becomes unwritable (full disk, Drive unmounted,
        permission error) is dropped without taking the run down.
    """

    def __init__(self, run_id=None, mirror_dirs=None, snapshot_every=100,
                 drive_flush_every=1):
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.snapshot_every = max(1, snapshot_every)
        self.drive_flush_every = max(1, drive_flush_every)
        self.state = {}
        self._current = None
        self._jsonl_handles = []
        self._steps_since_snapshot = 0
        self._snapshots_since_flush = 0

        candidate_dirs = list(mirror_dirs) if mirror_dirs is not None else list(LOSS_MIRROR_DIRS_DEFAULT)
        self.mirror_dirs = _resolve_mirror_dirs(candidate_dirs, "recorder")

        if not self.mirror_dirs:
            fallback = Path(tempfile.gettempdir()) / "midi_generator_losses_fallback"
            fallback.mkdir(parents=True, exist_ok=True)
            self.mirror_dirs.append(fallback)
            print(f"[recorder] all configured mirrors failed; using fallback {fallback}")

        print(f"[recorder] run_id={self.run_id} mirrors={[str(d) for d in self.mirror_dirs]} "
              f"snapshot_every={self.snapshot_every} drive_flush_every={self.drive_flush_every}")

    def start_model(self, genre, dataset_mode, model_name, meta, batch_size):
        entry = {
            "train_loss": [],
            "steps": 0,
            "meta": meta,
            "batch_size": batch_size,
            "started_at": datetime.now().isoformat(),
            "status": "running",
        }
        self.state.setdefault(genre, {}).setdefault(dataset_mode, {})[model_name] = entry
        self._current = (genre, dataset_mode, model_name)
        self._steps_since_snapshot = 0

        for h in self._jsonl_handles:
            try:
                h.close()
            except Exception:
                pass
        self._jsonl_handles = []

        stream_name = f"stream_{_safe_name(genre)}_{_safe_name(dataset_mode)}_{model_name}.jsonl"
        for d in self.mirror_dirs:
            try:
                self._jsonl_handles.append(open(d / stream_name, "a", buffering=1))
            except Exception as exc:  # noqa: BLE001
                print(f"[recorder] could not open stream in {d}: {exc}")

        self.snapshot()

    def record_step(self, step, loss):
        if self._current is None:
            return
        genre, dataset_mode, model_name = self._current
        entry = self.state[genre][dataset_mode][model_name]
        entry["train_loss"].append(float(loss))
        entry["steps"] = len(entry["train_loss"])

        line = json.dumps({"step": int(step), "loss": float(loss)}) + "\n"
        dead = []
        for h in self._jsonl_handles:
            try:
                h.write(line)
                h.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"[recorder] stream write failed, dropping handle: {exc}")
                dead.append(h)
        for h in dead:
            self._jsonl_handles.remove(h)
            try:
                h.close()
            except Exception:
                pass

        self._steps_since_snapshot += 1
        if self._steps_since_snapshot >= self.snapshot_every:
            self.snapshot()
            self._steps_since_snapshot = 0

    def end_model(self, status="done", error=None):
        if self._current is None:
            return
        genre, dataset_mode, model_name = self._current
        entry = self.state[genre][dataset_mode][model_name]
        entry["status"] = status
        entry["finished_at"] = datetime.now().isoformat()
        if error is not None:
            entry["error"] = str(error)

        for h in self._jsonl_handles:
            try:
                h.close()
            except Exception:
                pass
        self._jsonl_handles = []
        self._current = None
        self.snapshot(final=True)

    def snapshot(self, final=False, force_flush=False):
        payload = json.dumps(self.state, indent=2, default=str)
        live_name = f"losses_live_{self.run_id}.json"
        final_name = f"losses_final_{self.run_id}.json"
        targets = [live_name] + ([final_name] if final else [])

        for d in self.mirror_dirs:
            for name in targets:
                try:
                    _atomic_write_json(d / name, payload)
                except Exception as exc:  # noqa: BLE001
                    print(f"[recorder] snapshot to {d / name} failed: {exc}")

        self._snapshots_since_flush += 1
        if force_flush or final or self._snapshots_since_flush >= self.drive_flush_every:
            _flush_filesystem()
            self._snapshots_since_flush = 0

    def close(self):
        self.end_model(status="aborted") if self._current else self.snapshot(final=True)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_datasets(exp_path, chunk_size):
    """Create the (dataset_mode -> dataset) mapping for one genre."""
    return {
        "per_song": MidiDataset(
            Path(exp_path, "train"),
            preload_to_ram=True,
            chunk_size=chunk_size,
        ),
        "continuous": ContinuousMidiDataset(
            Path(exp_path, "train"),
            chunk_size=chunk_size,
        ),
    }


def run_all_training(genres, optimization_steps=10000, eval_interval=5000,
                     batch_size=64, chunk_size=CHUNK_SIZE,
                     loss_mirror_dirs=None, checkpoint_mirror_dirs=None,
                     snapshot_every=100, drive_flush_every=1, run_id=None):
    """Train every model config on every (genre, dataset) combination.

    Losses are mirrored to multiple destinations via `LossRecorder`, and each
    finished model's weights are also atomically written to every checkpoint
    mirror (local + Google Drive by default). Filesystem buffers are flushed
    after every checkpoint so that Colab's Drive FUSE driver actually pushes
    the data up to Google before we move on.

    Each model is wrapped in try/except so one failure does not discard
    progress from the other configurations.
    """
    print("Running all training processes...")
    recorder = LossRecorder(
        run_id=run_id,
        mirror_dirs=loss_mirror_dirs,
        snapshot_every=snapshot_every,
        drive_flush_every=drive_flush_every,
    )

    checkpoint_dirs = _resolve_mirror_dirs(
        list(checkpoint_mirror_dirs) if checkpoint_mirror_dirs is not None
        else list(CHECKPOINT_MIRROR_DIRS_DEFAULT),
        "checkpoint",
    )
    if not checkpoint_dirs:
        fallback = Path(tempfile.gettempdir()) / "midi_generator_checkpoints_fallback"
        fallback.mkdir(parents=True, exist_ok=True)
        checkpoint_dirs.append(fallback)
        print(f"[checkpoint] all configured mirrors failed; using fallback {fallback}")
    print(f"[checkpoint] mirrors={[str(d) for d in checkpoint_dirs]}")

    try:
        for genre_path in genres:
            print(f"\n=== Genre: {genre_path} ===")
            exp_path = Path("tokenization", "saved_tokens", genre_path)
            tokenizer = PerTok(params=Path(exp_path, "tokenizer.json"))
            vocab_size = tokenizer.vocab_size

            datasets = _build_datasets(exp_path, chunk_size)

            for dataset_mode, dataset in datasets.items():
                print(f"\n--- Dataset: {dataset_mode} ---")

                for cfg in build_model_configs(vocab_size, chunk_size):
                    print(f"\n>>> Training {cfg['name']}")
                    _reset_memory_stats()
                    free_memory(verbose=True)

                    cfg_batch_size = cfg.get("batch_size", batch_size)
                    meta = {
                        "needs_mask": cfg.get("needs_mask", False),
                        "needs_hidden": cfg.get("needs_hidden", False),
                        "stepwise": cfg.get("stepwise", False),
                        "use_amp": cfg.get("use_amp", False),
                        "grad_clip": cfg.get("grad_clip"),
                    }
                    recorder.start_model(
                        genre_path, dataset_mode, cfg["name"], meta, cfg_batch_size,
                    )

                    model = None
                    dataloader = None
                    try:
                        dataloader = torch.utils.data.DataLoader(
                            dataset, batch_size=cfg_batch_size, shuffle=True,
                        )
                        model = cfg["factory"]().to(device)
                        train_model(
                            model, dataloader, cfg, vocab_size,
                            optimization_steps, eval_interval=eval_interval,
                            recorder=recorder, checkpoint_mirror_dirs=checkpoint_dirs,
                        )
                        saved_paths = _save_final_checkpoint(
                            model, cfg["name"], genre_path, dataset_mode,
                            checkpoint_dirs,
                        )
                        entry = recorder.state.get(genre_path, {}).get(dataset_mode, {}).get(cfg["name"])
                        if entry is not None and saved_paths:
                            entry["checkpoint_paths"] = [str(p) for p in saved_paths]
                        recorder.end_model(status="done")

                    except KeyboardInterrupt:
                        print(f"[!] KeyboardInterrupt during {cfg['name']}")
                        recorder.end_model(status="interrupted", error="KeyboardInterrupt")
                        raise
                    except Exception as exc:  # noqa: BLE001
                        err = f"{type(exc).__name__}: {exc}"
                        print(f"[!] Training failed for {cfg['name']}: {err}")
                        traceback.print_exc()
                        recorder.end_model(status="failed", error=err)
                    finally:
                        if model is not None:
                            try:
                                model.to("cpu")
                            except Exception:
                                pass
                        del model, dataloader
                        free_memory(verbose=True)
                        _flush_filesystem()

    finally:
        recorder.close()
        _save_losses_snapshot(
            recorder.state, run_id or recorder.run_id,
            mirror_dirs=recorder.mirror_dirs,
        )
        _flush_filesystem()

    return recorder.state
