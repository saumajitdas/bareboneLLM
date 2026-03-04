import argparse
import torch

from .model import GPT
from .tokenizer import ByteTokenizer
from .utils import pick_device

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=64)
    args = ap.parse_args()

    device = pick_device("auto")
    tok = ByteTokenizer()

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
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

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.tensor(tok.encode(args.prompt), dtype=torch.long, device=device).unsqueeze(0)
    y = model.generate(
        x,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tok.decode(y[0].tolist()))

if __name__ == "__main__":
    main()
