from __future__ import annotations

from pathlib import Path
import sys
import urllib.request
import urllib.error

# Default: ~2GB TinyStories train split (as a single text file)
DEFAULT_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "TinyStoriesV2-GPT4-train.txt"
)

def download_with_progress(url: str, out_file: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """
    Stream-download a large file with progress output.
    Writes to <out_file>.part then renames to <out_file> atomically.
    """
    tmp = out_file.with_suffix(out_file.suffix + ".part")

    req = urllib.request.Request(url, headers={"User-Agent": "bareboneLLM/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total is not None else None

            downloaded = 0
            tmp.parent.mkdir(parents=True, exist_ok=True)

            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress
                    if total:
                        pct = downloaded * 100.0 / total
                        mb = downloaded / (1024 * 1024)
                        tmb = total / (1024 * 1024)
                        sys.stdout.write(f"\rDownloading... {pct:6.2f}%  ({mb:,.1f} / {tmb:,.1f} MB)")
                    else:
                        mb = downloaded / (1024 * 1024)
                        sys.stdout.write(f"\rDownloading... {mb:,.1f} MB")
                    sys.stdout.flush()

    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise SystemExit(f"\nHTTP error {e.code} downloading dataset: {e.reason}") from e
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise SystemExit(f"\nError downloading dataset: {e}") from e

    sys.stdout.write("\n")
    sys.stdout.flush()

    # Atomically replace/rename
    tmp.replace(out_file)

def main() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    out_file = data_dir / "train.txt"
    url = DEFAULT_URL

    if out_file.exists():
        size_gb = out_file.stat().st_size / (1024**3)
        print(f"Dataset already exists: {out_file} ({size_gb:.2f} GB)")
        print("Delete it if you want to re-download.")
        return

    print("Downloading dataset (this is large; may take a while depending on your internet)...")
    print(f"Source: {url}")
    download_with_progress(url, out_file)

    size_gb = out_file.stat().st_size / (1024**3)
    print(f"Saved dataset to: {out_file} ({size_gb:.2f} GB)")

if __name__ == "__main__":
    main()
