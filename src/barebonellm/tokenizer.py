from dataclasses import dataclass
from typing import List

@dataclass
class ByteTokenizer:
    vocab_size: int = 256  # fixed

    def encode(self, text: str) -> List[int]:
        b = text.encode("utf-8", errors="replace")
        return list(b)

    def decode(self, ids: List[int]) -> str:
        b = bytes([i % 256 for i in ids])
        return b.decode("utf-8", errors="replace")
