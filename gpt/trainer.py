import os
import time
from dataclasses import dataclass

import torch
from safetensors.torch import save_file, load_file


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

    # Hugging Face Hub
    push_to_hub: bool = False
    hf_repo_id: str = "ErnestBeckham/gpt-v1"
    hf_token: str = None  # falls back to a cached `huggingface-cli login` token / HF_TOKEN env var
    hf_private: bool = True


class Trainer:
    def __init__(self, model, train_data, val_data, config: TrainerConfig):
        self.model = model.to(config.device)
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.start_iter = 0
        self.best_val_loss = float("inf")

        self.model_path = os.path.join(config.checkpoint_dir, "model.safetensors")
        self.train_state_path = os.path.join(config.checkpoint_dir, "train_state.pt")

        self._hf_api = None
        if config.push_to_hub:
            self._hf_api = self._init_hf_repo()

        if config.resume and os.path.exists(self.model_path):
            self.load_checkpoint()

    def _init_hf_repo(self):
        from huggingface_hub import HfApi

        api = HfApi(token=self.config.hf_token)
        api.create_repo(repo_id=self.config.hf_repo_id, private=self.config.hf_private, exist_ok=True)
        return api

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        cfg = self.config
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
        return x.to(cfg.device), y.to(cfg.device)

    @torch.no_grad()
    def estimate_loss(self):
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
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        state_dict = {k: v.contiguous() for k, v in self.model.state_dict().items()}
        save_file(state_dict, self.model_path, metadata={"iter": str(iter_num), "val_loss": str(val_loss)})

        torch.save({
            "optimizer_state": self.optimizer.state_dict(),
            "iter": iter_num,
            "best_val_loss": val_loss,
        }, self.train_state_path)

        if self.config.push_to_hub:
            self.push_checkpoint(iter_num, val_loss)

    def load_checkpoint(self):
        state_dict = load_file(self.model_path, device=self.config.device)
        self.model.load_state_dict(state_dict)

        if os.path.exists(self.train_state_path):
            train_state = torch.load(self.train_state_path, map_location=self.config.device)
            self.optimizer.load_state_dict(train_state["optimizer_state"])
            self.start_iter = train_state.get("iter", 0) + 1
            self.best_val_loss = train_state.get("best_val_loss", float("inf"))

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

        for iter_num in range(self.start_iter, cfg.max_iters):
            if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
                losses = self.estimate_loss()
                elapsed = time.time() - start_time
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                      f"elapsed {elapsed:.1f}s")

                if losses["val"] < self.best_val_loss:
                    self.best_val_loss = losses["val"]
                    self.save_checkpoint(iter_num, self.best_val_loss)

            xb, yb = self.get_batch("train")
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

        return self.model
