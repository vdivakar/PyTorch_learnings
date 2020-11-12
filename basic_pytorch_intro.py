import torch
import numpy as np

data = np.array([1,2,3,4])

t1 = torch.Tensor(data)
t2 = torch.tensor(data)

print(t1.dtype)
print(t2.dtype)
