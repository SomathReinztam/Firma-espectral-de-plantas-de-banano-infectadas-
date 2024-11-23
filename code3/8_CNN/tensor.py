"""
tensor.py

"""

import numpy as np
import torch


x = np.array([1.0, 1.0, 1.0])
x = torch.from_numpy(x).view(1, 1, -1)

print(x)
print(x.shape)
print(x.size())



