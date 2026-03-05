import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import TokenDataset
from .model import GPT
from .set_seed import set_seed
from .tokenizer import ByteTokenizer
from .utils import ensure_dir, pick_device

from .streaming_dataset import StreamTextDataset

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(cfg.get("seed", 1337)))

    device = pick_device(cfg.get("device", "auto"))

    use_streaming = bool(cfg.get("streaming", False))

    if use_streaming:
      ds = StreamTextDataset(
          path=cfg["data_path"],
          context_length=int(cfg["context_length"]),
          chunk_bytes=int(cfg.get("chunk_bytes", 4 * 1024 * 1024)),
          stride=int(cfg.get("stride", 8)),
      )

      dl = DataLoader(
          ds,
          batch_size=int(cfg["batch_size"]),
          shuffle=False,
          num_workers=int(cfg.get("num_workers", 0)),
      )

    else:
      text = Path(cfg["data_path"]).read_text(encoding="utf-8", errors="replace")
      tok = ByteTokenizer()
      ids = torch.tensor(tok.encode(text), dtype=torch.long)

      ds = TokenDataset(ids, context_length=int(cfg["context_length"]))

      dl = DataLoader(
          ds,
          batch_size=int(cfg["batch_size"]),
          shuffle=True,
          num_workers=int(cfg.get("num_workers", 0)),
      )


    mcfg = cfg["model"]
    model = GPT(
        vocab_size=int(mcfg["vocab_size"]),
        context_length=int(cfg["context_length"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        d_model=int(mcfg["d_model"]),
        d_ff=int(mcfg["d_ff"]),
        dropout=float(mcfg.get("dropout", 0.0)),
    ).to(device)

    tcfg = cfg["train"]
    ckpt_dir = tcfg.get("checkpoint_dir", "checkpoints")
    ensure_dir(ckpt_dir)
    ckpt_path = Path(ckpt_dir) / "model.pt"

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    max_steps = int(tcfg["max_steps"])
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    log_every = int(tcfg.get("log_every", 50))
    save_every = int(tcfg.get("save_every", 500))

    it = iter(dl)
    pbar = tqdm(total=max_steps, desc="train")
    step = 0

    while step < max_steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl)
            x, y = next(it)

        x = x.to(device)
        y = y.to(device)

        model.train()
        _, loss = model(x, y)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()

        step += 1
        pbar.update(1)

        if step % log_every == 0:
            pbar.set_postfix(loss=float(loss.detach().cpu()))

        if step % save_every == 0 or step == max_steps:
            torch.save({"config": cfg, "model_state": model.state_dict()}, ckpt_path)

    pbar.close()
    print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
