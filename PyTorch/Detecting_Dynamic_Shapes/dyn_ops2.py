# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.

from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model
import torch


class InnerNet(torch.nn.Module):
   def __init__(self):
      super(InnerNet, self).__init__()
      self.conv = torch.nn.Conv2d(1, 8, 3, 3)

   def forward(self, x):
      x = torch.flatten(self.conv(x), 1)
      #x = x[x>0] # This is dynamic, replacing in next line with static implementation
      x = torch.where(x>0, x, torch.zeros_like(x))
      return x.sum()

net = torch.nn.Sequential(torch.nn.ReLU(), InnerNet()).to('hpu')
net = detect_recompilation_auto_model(net)

for bs in [20,20,30,30]: #Input shape changes at 4th step
   inp = torch.rand(bs, 1, 50, 50).to('hpu')
   print(net(inp))
net.analyse_dynamicity() # Call this after a few steps to generate the dynamicity report
