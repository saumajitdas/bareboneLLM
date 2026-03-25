from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TokenDataset
from .model import GPT
from .set_seed import set_seed
from .streaming_dataset import StreamTokenizedDataset
from .tokenizer import build_tokenizer
from .utils import ensure_dir, pick_device


@torch.no_grad()
def evaluate(
    model: GPT,
    dl: DataLoader,
    device: str,
    max_eval_steps: int = 50,
) -> tuple[Optional[float], Optional[float]]:
    model.eval()
    losses = []

    it = iter(dl)
    for _ in range(max_eval_steps):
        try:
            x, y = next(it)
        except StopIteration:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        losses.append(loss.item())

    if not losses:
        return None, None

    val_loss = sum(losses) / len(losses)
    ppl = math.exp(val_loss)
    return val_loss, ppl


def build_dataloader(
    *,
    data_path: str,
    tokenizer_cfg: dict,
    context_length: int,
    batch_size: int,
    num_workers: int,
    use_streaming: bool,
    chunk_bytes: int,
    stride: int,
    loop: bool,
    shuffle_non_streaming: bool,
) -> DataLoader:
    if use_streaming:
        ds = StreamTokenizedDataset(
            path=data_path,
            tokenizer_cfg=tokenizer_cfg,
            context_length=context_length,
            chunk_bytes=chunk_bytes,
            stride=stride,
            loop=loop,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    tokenizer = build_tokenizer(tokenizer_cfg)
    text = Path(data_path).read_text(encoding="utf-8", errors="replace")
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    ds = TokenDataset(ids, context_length=context_length)

    if len(ds) <= 0:
        raise ValueError(
            f"Dataset too small for context_length={context_length}. File: {data_path}"
        )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle_non_streaming,
        num_workers=num_workers,
    )


def save_checkpoint(
    *,
    path: Path,
    cfg: dict,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: Optional[float],
) -> None:
    payload = {
        "config": cfg,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 1337)))

    device = pick_device(cfg.get("device", "auto"))

    tokenizer_cfg = cfg.get("tokenizer", {"type": "byte"})
    tokenizer = build_tokenizer(tokenizer_cfg)

    context_length = int(cfg["context_length"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))

    use_streaming = bool(cfg.get("streaming", False))
    chunk_bytes = int(cfg.get("chunk_bytes", 4 * 1024 * 1024))
    stride = int(cfg.get("stride", 8))
    loop = bool(cfg.get("loop", True))

    train_dl = build_dataloader(
        data_path=cfg["data_path"],
        tokenizer_cfg=tokenizer_cfg,
        context_length=context_length,
        batch_size=batch_size,
        num_workers=num_workers,
        use_streaming=use_streaming,
        chunk_bytes=chunk_bytes,
        stride=stride,
        loop=loop,
        shuffle_non_streaming=True,
    )

    val_dl = None
    val_data_path = cfg.get("val_data_path")
    if val_data_path:
        val_streaming = bool(cfg.get("val_streaming", use_streaming))
        val_chunk_bytes = int(cfg.get("val_chunk_bytes", chunk_bytes))
        val_stride = int(cfg.get("val_stride", stride))
        val_loop = bool(cfg.get("val_loop", False))

        val_dl = build_dataloader(
            data_path=val_data_path,
            tokenizer_cfg=tokenizer_cfg,
            context_length=context_length,
            batch_size=batch_size,
            num_workers=num_workers if val_streaming else 0,
            use_streaming=val_streaming,
            chunk_bytes=val_chunk_bytes,
            stride=val_stride,
            loop=val_loop,
            shuffle_non_streaming=False,
        )

    mcfg = cfg["model"]
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        context_length=context_length,
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        d_model=int(mcfg["d_model"]),
        d_ff=int(mcfg["d_ff"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    ).to(device)

    tcfg = cfg["train"]
    ckpt_dir = Path(tcfg.get("checkpoint_dir", "checkpoints"))
    ensure_dir(str(ckpt_dir))

    last_ckpt_path = ckpt_dir / "model.pt"
    best_ckpt_path = ckpt_dir / "best_model.pt"

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    max_steps = int(tcfg["max_steps"])
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 500))

    eval_every = int(cfg.get("eval_every", 0))
    eval_steps = int(cfg.get("eval_steps", 50))

    # Resume support
    step = 0
    best_val_loss: Optional[float] = None
    resume_from = cfg.get("resume_from")

    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.exists():
            print(f"Resuming from checkpoint: {resume_path}")
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            step = int(ckpt.get("step", 0))
            best_val_loss = ckpt.get("best_val_loss")
            print(f"Resumed at step={step}, best_val_loss={best_val_loss}")
        else:
            print(f"resume_from not found: {resume_path}. Starting fresh.")

    train_it = iter(train_dl)
    pbar = tqdm(total=max_steps, initial=step, desc="train")

    while step < max_steps:
        try:
            x, y = next(train_it)
        except StopIteration:
            train_it = iter(train_dl)
            x, y = next(train_it)

        x = x.to(device)
        y = y.to(device)

        model.train()
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        step += 1
        pbar.update(1)

        if step % log_every == 0:
            pbar.set_postfix(train_loss=float(loss.detach().cpu()))

        if val_dl is not None and eval_every > 0 and step % eval_every == 0:
            val_loss, ppl = evaluate(
                model=model,
                dl=val_dl,
                device=device,
                max_eval_steps=eval_steps,
            )
            if val_loss is not None:
                print(
                    f"\nstep {step} | "
                    f"train_loss={float(loss.detach().cpu()):.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"ppl={ppl:.2f}"
                )

                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        path=best_ckpt_path,
                        cfg=cfg,
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        best_val_loss=best_val_loss,
                    )
                    print(
                        f"New best checkpoint saved to {best_ckpt_path} "
                        f"(val_loss={best_val_loss:.4f})"
                    )

        if step % save_every == 0 or step == max_steps:
            save_checkpoint(
                path=last_ckpt_path,
                cfg=cfg,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=best_val_loss,
            )

    pbar.close()
    print(f"Saved latest checkpoint to {last_ckpt_path}")
    if best_val_loss is not None:
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()