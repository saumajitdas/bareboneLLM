from dataclasses import dataclass
from typing import Tuple
import torch
from torch.utils.data import Dataset

@dataclass
class TokenDataset(Dataset):
    tokens: torch.Tensor  # 1D long tensor
    context_length: int

    def __len__(self) -> int:
        return max(0, self.tokens.numel() - self.context_length - 1)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[i : i + self.context_length]
        y = self.tokens[i + 1 : i + 1 + self.context_length]
        return x, y
