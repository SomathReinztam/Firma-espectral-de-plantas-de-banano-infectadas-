"""
1_Conv1d.py

convoluciones 1d y 2d:
https://chatgpt.com/share/672ad381-e228-8013-a8cb-a1bf204c38a2
"""

import torch 
import torch.nn as nn

import numpy as np

x = np.linspace(0, 10, 15)
y = np.sin(x)

print(y)
print(y.shape)
print()

y = torch.tensor(y, dtype=torch.float32).view(1, 1, -1)

print(y)
print(y.shape)
print()

m = nn.Conv1d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=2,
    padding=1
)


y = m(y)

print(y)
print(y.shape)
print()