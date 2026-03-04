from pathlib import Path

SAMPLE = """\
This is a tiny sample dataset.
Replace this with real text for meaningful training.
The quick brown fox jumps over the lazy dog.
"""

def main() -> None:
    Path("data").mkdir(parents=True, exist_ok=True)
    p = Path("data/train.txt")
    if not p.exists():
        p.write_text(SAMPLE, encoding="utf-8")
        print(f"Wrote {p}")
    else:
        print(f"{p} already exists")

if __name__ == "__main__":
    main()
