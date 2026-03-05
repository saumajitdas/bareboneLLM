import os
from dataclasses import dataclass
from typing import Iterator, Tuple

import torch
from torch.utils.data import IterableDataset


@dataclass
class StreamTextDataset(IterableDataset):
    """
    Stream a large file as byte tokens (0..255) and yield (x,y) windows.
    Works with your existing ByteTokenizer logic because bytes ARE the tokens.
    """
    path: str
    context_length: int
    chunk_bytes: int = 4 * 1024 * 1024  # 4MB
    stride: int = 8                      # >1 reduces overlap => faster
    loop: bool = True                    # keep cycling through file

    def _iter_range(self, start: int, end: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        with open(self.path, "rb") as f:
            f.seek(start)
            remaining = end - start
            buffer = bytearray()

            while True:
                if remaining <= 0:
                    if not self.loop:
                        break
                    # loop back to start of assigned range
                    f.seek(start)
                    remaining = end - start

                to_read = min(self.chunk_bytes, remaining)
                chunk = f.read(to_read)
                if not chunk:
                    if not self.loop:
                        break
                    f.seek(start)
                    remaining = end - start
                    continue

                remaining -= len(chunk)
                buffer.extend(chunk)

                # Need context_length + 1 bytes to form x and y
                while len(buffer) >= self.context_length + 1:
                    window = buffer[: self.context_length + 1]
                    t = torch.tensor(list(window), dtype=torch.long)
                    x = t[:-1]
                    y = t[1:]
                    yield x, y
                    del buffer[: self.stride]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        file_size = os.path.getsize(self.path)

        # Multi-worker split: each worker reads a disjoint contiguous region
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, file_size
        else:
            n = worker_info.num_workers
            wid = worker_info.id
            per = file_size // n
            start = wid * per
            end = file_size if wid == n - 1 else (wid + 1) * per

            # back up a bit so boundaries still have enough context
            start = max(0, start - (self.context_length + 1))

        return self._iter_range(start, end)
