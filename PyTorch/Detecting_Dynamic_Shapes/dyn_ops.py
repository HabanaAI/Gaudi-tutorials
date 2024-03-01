# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.

import torch
from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model


class InnerNet(torch.nn.Module):
    def __init__(self):
        super(InnerNet, self).__init__()
        self.conv = torch.nn.Conv2d(1, 8, 3, 3)

    def forward(self, x):
        x = torch.flatten(self.conv(x), 1)
        x = x[x > 0]  # This is dynamic
        return x.sum()


net = torch.nn.Sequential(torch.nn.ReLU(), InnerNet()).to("hpu")
net = detect_recompilation_auto_model(net)  # wrap model in dynamic op detection tool

for bs in [20, 20, 30, 30]:  # Input shape changes at 3rd step
    inp = torch.rand(bs, 1, 50, 50).to("hpu")
    print(net(inp))
net.analyse_dynamicity()  # Call this after a few steps to generate the dynamicity report
