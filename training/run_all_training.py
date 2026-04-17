"""Unified training orchestration for all model families.

Exposes a single `train_model` loop that handles GRU and Transformer variants
via a lightweight config/adapter scheme, plus a `run_all_training` driver that
iterates over genres, dataset modes, and model configs and collects losses
into a nested dictionary.
"""

import json
import time
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

    return loss.item()


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

    return sum(token_losses) / max(1, len(token_losses))


# ---------------------------------------------------------------------------
# Unified training loop
# ---------------------------------------------------------------------------

def train_model(model, dataloader, cfg, vocab_size, n_steps, eval_interval=1000,
                checkpoint_seconds=CHECKPOINT_INTERVAL_SECONDS):
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

        if scheduler is not None:
            scheduler.step()

        if (step + 1) % eval_interval == 0:
            print(f"\n[{cfg['name']}] Step {step + 1} | Loss {step_loss:.4f}")

        if time.time() - start_time >= checkpoint_seconds:
            _save_intermediate_checkpoint(model, optimizer, cfg["name"], step)
            start_time = time.time()

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
        },
    ]


# ---------------------------------------------------------------------------
# Checkpoint / loss persistence helpers
# ---------------------------------------------------------------------------

def _save_intermediate_checkpoint(model, optimizer, name, step):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"{name}_model_{ts}_{step}")
    torch.save(optimizer.state_dict(), CHECKPOINT_DIR / f"{name}_optimizer_{ts}_{step}")


def _save_final_checkpoint(model, name, genre, dataset_mode):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_genre = str(genre).replace("/", "_").replace("\\", "_")
    filename = f"{name}_{dataset_mode}_{safe_genre}_{ts}"
    path = CHECKPOINT_DIR / filename
    torch.save(model.state_dict(), path)
    if _HAS_COLAB:
        try:
            colab_files.download(str(path))
        except Exception as exc:  # noqa: BLE001 - best-effort download
            print(f"Colab download skipped for {path.name}: {exc}")
    return path


def _save_losses_snapshot(losses_dict, tag):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = str(tag).replace("/", "_").replace("\\", "_")
    path = CHECKPOINT_DIR / f"losses_{safe_tag}_{ts}.json"
    with open(path, "w") as f:
        json.dump(losses_dict, f, indent=2)
    return path


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
                     batch_size=64, chunk_size=CHUNK_SIZE):
    """Train every model config on every (genre, dataset) combination.

    Returns a nested dictionary:
        losses[genre][dataset_mode][model_name] = {
            "train_loss": [...],
            "steps": int,
            "meta": {...},
        }
    """
    print("Running all training processes...")
    all_losses = {}

    for genre_path in genres:
        print(f"\n=== Genre: {genre_path} ===")
        exp_path = Path("tokenization", "saved_tokens", genre_path)
        tokenizer = PerTok(params=Path(exp_path, "tokenizer.json"))
        vocab_size = tokenizer.vocab_size

        datasets = _build_datasets(exp_path, chunk_size)
        all_losses[genre_path] = {}

        for dataset_mode, dataset in datasets.items():
            print(f"\n--- Dataset: {dataset_mode} ---")
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True,
            )
            all_losses[genre_path][dataset_mode] = {}

            for cfg in build_model_configs(vocab_size, chunk_size):
                print(f"\n>>> Training {cfg['name']}")
                model = cfg["factory"]().to(device)

                losses = train_model(
                    model, dataloader, cfg, vocab_size,
                    optimization_steps, eval_interval=eval_interval,
                )
                _save_final_checkpoint(model, cfg["name"], genre_path, dataset_mode)

                all_losses[genre_path][dataset_mode][cfg["name"]] = {
                    "train_loss": losses,
                    "steps": len(losses),
                    "meta": {
                        "needs_mask": cfg.get("needs_mask", False),
                        "needs_hidden": cfg.get("needs_hidden", False),
                        "stepwise": cfg.get("stepwise", False),
                        "use_amp": cfg.get("use_amp", False),
                        "grad_clip": cfg.get("grad_clip"),
                    },
                }

                _save_losses_snapshot(all_losses, genre_path)

                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    return all_losses
