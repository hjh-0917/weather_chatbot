# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OutfitNet(nn.Module):
    """
    간단한 fully-connected 네트워크.
    입력: categorical features를 one-hot / encoded 한 벡터 (사이즈 input_dim)
    출력: 상의/하의/신발/액세서리 각각의 분류 logits (멀티헤드)
    """
    def __init__(self, input_dim, hidden_dim, head_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # head_sizes: dict like {"top": n_top_classes, "bottom": n_bottom_classes, ...}
        self.heads = nn.ModuleDict()
        for k, sz in head_sizes.items():
            self.heads[k] = nn.Linear(hidden_dim, sz)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = {}
        for k, layer in self.heads.items():
            out[k] = layer(x)  # raw logits
        return out
