import os
from functools import partial
import torchvision.transforms.functional as F

dir_path = os.path.dirname(os.path.realpath(__file__))

def rotate(raw_image, severity, attribute_label):
    if severity==0:
        raise NotImplementedError("Need severity != 0")
    rotation = 90/(5-severity)
    if attribute_label == 0:
        return raw_image
    elif attribute_label == 1:
        image = F.rotate(raw_image.unsqueeze(0).float(),rotation).squeeze(0)
        return image
    else: raise NotImplementedError("Only 2class-dataset")

ROTATED_MNIST_PROTOCOL = dict()
for i in range(2):
    ROTATED_MNIST_PROTOCOL[i] = partial(rotate, attribute_label = i)