import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import save_file, load_file

import metrics


@dataclass
class TrainerConfig:
    max_iters: int = 6000
    eval_interval: int = 500
    eval_iters: int = 200
    batch_size: int = 4
    block_size: int = 8
    learning_rate: float = 3e-4
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "../models"
    resume: bool = False

    # Phase 3: set when launched with torchrun (checked via env, see is_ddp())
    ddp: bool = False

    # Phase 7: expose /metrics on rank 0 for Prometheus to scrape
    metrics_port: int = 8000

    # Hugging Face Hub
    push_to_hub: bool = False
    hf_repo_id: str = "ErnestBeckham/gpt-v1"
    hf_token: str = None  # falls back to a cached `huggingface-cli login` token / HF_TOKEN env var
    hf_private: bool = True


def is_ddp():
    """torchrun sets this; a plain `python train_gpt.py` run does not."""
    return "RANK" in os.environ


def resolve_device():
    """Call before constructing the model, so module-level globals like
    gpt_v1.device get the right per-process GPU, not just "cuda" (= GPU 0
    for every rank)."""
    if is_ddp():
        return f"cuda:{os.environ['LOCAL_RANK']}"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, model, train_data, val_data, config: TrainerConfig):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.start_iter = 0
        self.best_val_loss = float("inf")

        # --- Phase 3: DDP setup -- one process per GPU, NCCL for the sync ---
        config.ddp = is_ddp()
        if config.ddp:
            dist.init_process_group(backend="nccl")
            self.rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = dist.get_world_size()
            config.device = resolve_device()
            torch.cuda.set_device(config.device)
        else:
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
        self.is_main = self.rank == 0

        self.model_path = os.path.join(config.checkpoint_dir, "model.safetensors")
        self.train_state_path = os.path.join(config.checkpoint_dir, "train_state.pt")

        if config.resume and os.path.exists(self.model_path):
            self.load_checkpoint(model)

        model = model.to(config.device)
        self.raw_model = model  # unwrapped -- used for checkpointing
        if config.ddp:
            self.model = DDP(model, device_ids=[self.local_rank])
        else:
            self.model = model

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        if config.resume and os.path.exists(self.train_state_path):
            self.load_train_state()

        self._hf_api = None
        if config.push_to_hub and self.is_main:
            self._hf_api = self._init_hf_repo()

        if self.is_main:
            metrics.start(config.metrics_port)

    def _init_hf_repo(self):
        from huggingface_hub import HfApi

        api = HfApi(token=self.config.hf_token)
        api.create_repo(repo_id=self.config.hf_repo_id, private=self.config.hf_private, exist_ok=True)
        return api

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        cfg = self.config
        # Each rank draws its own random batch (different generator state per
        # process, since torch's default seed is per-process) -- simplest way
        # to avoid every GPU training on identical data without sharding files.
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

    @torch.no_grad()
    def estimate_loss(self):
        """Only rank 0 evaluates -- a DDP forward pass with no backward needs
        no cross-rank sync, so this doesn't block the other ranks."""
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                xb, yb = self.get_batch(split)
                _, loss = self.model(xb, yb)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def save_checkpoint(self, iter_num, val_loss):
        """Phase 4: always save (not just on improvement) so a killed run can
        resume from its latest step, with RNG state for reproducibility."""
        if not self.is_main:
            return
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        state_dict = {k: v.contiguous() for k, v in self.raw_model.state_dict().items()}
        save_file(state_dict, self.model_path, metadata={"iter": str(iter_num), "val_loss": str(val_loss)})

        torch.save({
            "optimizer_state": self.optimizer.state_dict(),
            "iter": iter_num,
            "best_val_loss": self.best_val_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, self.train_state_path)

        if self.config.push_to_hub:
            self.push_checkpoint(iter_num, val_loss)

    def load_checkpoint(self, model):
        """Load model weights only; called before DDP-wrapping the model."""
        state_dict = load_file(self.model_path, device="cpu")
        model.load_state_dict(state_dict)

    def load_train_state(self):
        train_state = torch.load(self.train_state_path, map_location=self.config.device)
        self.optimizer.load_state_dict(train_state["optimizer_state"])
        self.start_iter = train_state.get("iter", 0) + 1
        self.best_val_loss = train_state.get("best_val_loss", float("inf"))

        torch.set_rng_state(train_state["rng_state"].cpu())
        if train_state.get("cuda_rng_state") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(train_state["cuda_rng_state"])

        if self.is_main:
            print(f"Resumed from checkpoint at iter {self.start_iter}, best val loss {self.best_val_loss:.4f}")

    def push_checkpoint(self, iter_num, val_loss):
        self._hf_api.upload_file(
            path_or_fileobj=self.model_path,
            path_in_repo="model.safetensors",
            repo_id=self.config.hf_repo_id,
            commit_message=f"checkpoint at iter {iter_num}, val loss {val_loss:.4f}",
        )

    def train(self):
        cfg = self.config
        start_time = time.time()
        step_time = start_time

        for iter_num in range(self.start_iter, cfg.max_iters):
            if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
                if self.is_main:
                    losses = self.estimate_loss()
                    elapsed = time.time() - start_time
                    print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                          f"elapsed {elapsed:.1f}s")

                    self.best_val_loss = min(self.best_val_loss, losses["val"])
                    metrics.step.set(iter_num)
                    metrics.loss_train.set(losses["train"])
                    metrics.loss_val.set(losses["val"])
                    self.save_checkpoint(iter_num, losses["val"])

            xb, yb = self.get_batch("train")
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if self.is_main:
                now = time.time()
                metrics.step_seconds.set(now - step_time)
                metrics.tokens_per_sec.set(cfg.batch_size * cfg.block_size * self.world_size / (now - step_time))
                step_time = now

        if cfg.ddp:
            dist.destroy_process_group()
        return self.raw_model
