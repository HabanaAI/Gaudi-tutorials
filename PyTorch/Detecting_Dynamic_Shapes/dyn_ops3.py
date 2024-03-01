# Copyright (c) 2023, Habana Labs Ltd.  All rights reserved.

import os

import habana_frameworks.torch.core as htcore
import torchvision
import torchvision.transforms as T
from PIL import Image

device = "hpu"

# load model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # set to evaluation mode
model = model.to(device)  # move model to device

from habana_frameworks.torch.utils.experimental import detect_recompilation_auto_model  # noqa: E402

model = detect_recompilation_auto_model(model, waittime=0.3)

for idx, k in enumerate(os.listdir("coco128/images/train2017/")):
    img = Image.open("coco128/images/train2017/" + k).resize((600, 600))
    img = T.ToTensor()(img).to(device)
    print("inp shape:", img.shape)
    pred = model([img])
    htcore.mark_step()
    if idx == 6:  # just running first few images
        break
    print("done img", idx)
model.analyse_dynamicity()
