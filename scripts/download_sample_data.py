from __future__ import annotations

from pathlib import Path
import random
import urllib.request

from datasets import load_dataset


TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)

# Hugging Face Wikipedia dataset config
WIKI_DATASET_NAME = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.en"


def download_tinystories(path: Path) -> None:
    if path.exists():
        print(f"TinyStories file already exists: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading TinyStories V2 GPT-4...")
    urllib.request.urlretrieve(TINYSTORIES_URL, path)
    print(f"Saved TinyStories to {path} ({path.stat().st_size / (1024**3):.2f} GB)")


def write_mixed_corpus(
    *,
    tiny_path: Path,
    train_out: Path,
    val_out: Path,
    total_target_bytes: int = 2_000_000_000,   # ~2GB total corpus
    val_fraction: float = 0.005,               # 0.5% validation
    wiki_ratio: float = 0.70,                  # 70% Wikipedia
    tiny_ratio: float = 0.30,                  # 30% TinyStories
    seed: int = 1337,
) -> None:
    assert abs((wiki_ratio + tiny_ratio) - 1.0) < 1e-9

    rng = random.Random(seed)

    train_out.parent.mkdir(parents=True, exist_ok=True)

    # Read TinyStories once; it is a local flat text file.
    tiny_text = tiny_path.read_text(encoding="utf-8", errors="ignore")
    tiny_len = len(tiny_text)
    tiny_idx = 0

    # Wikipedia streaming iterator from Hugging Face.
    wiki_iter = iter(
        load_dataset(
            WIKI_DATASET_NAME,
            WIKI_CONFIG,
            split="train",
            streaming=True,
        )
    )

    target_train_bytes = int(total_target_bytes * (1.0 - val_fraction))
    target_val_bytes = total_target_bytes - target_train_bytes

    written_train = 0
    written_val = 0

    print("Building mixed corpus...")
    print(f"Target total size: {total_target_bytes / (1024**3):.2f} GB")
    print(f"Train target:      {target_train_bytes / (1024**3):.2f} GB")
    print(f"Val target:        {target_val_bytes / (1024**3):.2f} GB")
    print(f"Mix:               {wiki_ratio:.0%} Wikipedia / {tiny_ratio:.0%} TinyStories")

    with open(train_out, "w", encoding="utf-8") as f_train, open(
        val_out, "w", encoding="utf-8"
    ) as f_val:
        while written_train < target_train_bytes or written_val < target_val_bytes:
            use_wiki = rng.random() < wiki_ratio

            if use_wiki:
                try:
                    sample = next(wiki_iter)
                except StopIteration:
                    wiki_iter = iter(
                        load_dataset(
                            WIKI_DATASET_NAME,
                            WIKI_CONFIG,
                            split="train",
                            streaming=True,
                        )
                    )
                    sample = next(wiki_iter)

                text = sample.get("text", "")
                title = sample.get("title", "")

                if not text:
                    continue

                record = f"{title}\n{text}\n\n"
            else:
                # Pull a random-ish chunk from TinyStories to avoid a single fixed pass.
                if tiny_idx >= tiny_len:
                    tiny_idx = 0

                chunk_size = 4000
                start = tiny_idx
                end = min(tiny_len, start + chunk_size)
                record = tiny_text[start:end].strip() + "\n\n"
                tiny_idx = end

                if len(record.strip()) < 20:
                    continue

            record_bytes = len(record.encode("utf-8"))

            # Route into train or val split
            if written_val < target_val_bytes and rng.random() < val_fraction:
                f_val.write(record)
                written_val += record_bytes
            else:
                if written_train < target_train_bytes:
                    f_train.write(record)
                    written_train += record_bytes
                elif written_val < target_val_bytes:
                    f_val.write(record)
                    written_val += record_bytes

            if (written_train + written_val) % 100_000_000 < record_bytes:
                total_written = written_train + written_val
                print(
                    f"Progress: {total_written / (1024**3):.2f} GB "
                    f"(train={written_train / (1024**3):.2f} GB, "
                    f"val={written_val / (1024**3):.2f} GB)"
                )

    print("Done.")
    print(f"Train file: {train_out} ({train_out.stat().st_size / (1024**3):.2f} GB)")
    print(f"Val file:   {val_out} ({val_out.stat().st_size / (1024**3):.2f} GB)")


def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    tiny_path = data_dir / "tinystories.txt"
    train_out = data_dir / "train.txt"
    val_out = data_dir / "val.txt"

    download_tinystories(tiny_path)

    write_mixed_corpus(
        tiny_path=tiny_path,
        train_out=train_out,
        val_out=val_out,
        total_target_bytes=2_000_000_000,
        val_fraction=0.005,
        wiki_ratio=0.70,
        tiny_ratio=0.30,
        seed=1337,
    )


if __name__ == "__main__":
    main()