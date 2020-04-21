import torch
import numpy as numpy


cityscape_cp = torch.load("dla34up-cityscapes-ed5cc4e8.pth")
# print(list(cityscape_cp.keys()))
print(cityscape_cp['fc.0.bias'].shape)