import os
from functools import partial
import torchvision.transforms.functional as F

dir_path = os.path.dirname(os.path.realpath(__file__))

def shift(raw_image, severity, attribute_label):
    if severity==0:
        raise NotImplementedError("Need severity != 0")
    translation = 8/(5-severity)
    if attribute_label == 0:
        image = F.affine(raw_image.unsqueeze(0).float(),scale=1,shear=0,angle=0,translate=(translation,translation/2)).squeeze(0)
        return image
    elif attribute_label == 1:
        image = F.affine(raw_image.unsqueeze(0).float(),scale=1,shear=0,angle=0,translate=(-translation,-translation/2)).squeeze(0)
        return image
    else: raise NotImplementedError("Only 2class-dataset")

SHIFTED_MNIST_PROTOCOL = dict()
for i in range(2):
    SHIFTED_MNIST_PROTOCOL[i] = partial(shift, attribute_label = i)

