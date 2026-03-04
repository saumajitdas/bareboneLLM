import argparse
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .model import GPT
from .tokenizer import ByteTokenizer
from .utils import pick_device

app = FastAPI(title="bareboneLLM")

TOK = ByteTokenizer()
DEVICE = pick_device("auto")
MODEL: GPT | None = None

class GenRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = 64

class GenResponse(BaseModel):
    text: str

@app.post("/generate", response_model=GenResponse)
def generate(req: GenRequest) -> GenResponse:
    assert MODEL is not None, "Model not loaded"
    ids = TOK.encode(req.prompt)
    x = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    y = MODEL.generate(
        x,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
    )
    return GenResponse(text=TOK.decode(y[0].tolist()))

def load_model(ckpt_path: str) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
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
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    global MODEL
    MODEL = load_model(args.checkpoint)

    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
