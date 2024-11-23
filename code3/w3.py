"""
import sklearn
print(sklearn.__version__)
"""

import torch

x = torch.linspace(0, 10, 20)
y_s = torch.sin(x)
y_c = torch.cos(x)

y = torch.stack((y_s, y_c), dim=1)

y = y.unsqueeze(0)

y = y.permute(0, 2, 1)

print(y.shape)

#print(y[0].permute(1, 0))