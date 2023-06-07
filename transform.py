import numpy as np
import torch
import cv2
import torchvision

class Reshape(object):
    '''
    Reshape input tensor from (X, 3, H, W) to (X, H, W, 3)
    '''
    def __init__(self):
        pass

    def __call__(self, im):
        print("reshaping: ",im.shape)
        return im.transpose(0, 2, 3, 1)


class PadandRandomCrop(object):
    '''
    Input tensor is expected to have shape of (H, W, 3)
    '''
    def __init__(self, border=4, cropsize=(32, 32)):
        self.border = border
        self.cropsize = cropsize

    def __call__(self, im):
        print("padding: ", im.shape)
        borders = [(self.border, self.border), (self.border, self.border), (0, 0)]
        convas = np.pad(im, borders, mode='reflect')
        H, W, C = convas.shape
        h, w = self.cropsize
        dh, dw = max(0, H-h), max(0, W-w)
        sh, sw = np.random.randint(0, dh), np.random.randint(0, dw)
        out = convas[sh:sh+h, sw:sw+w, :]
        return out


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im):
        print("flipping: ", im.shape)
        if np.random.rand() < self.p:
            im = im[:, ::-1, :]
        return im


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, im):
        im = cv2.resize(im, self.size)
        return im


class Normalize(torch.nn.Module):
    '''
    Inputs are pixel values in range of [0, 255], channel order is 'rgb'
    '''
    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return torch.nn.functional.normalize(tensor, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, im):
        if len(im.shape) == 4:
            return torch.from_numpy(im.transpose(0, 3, 1, 2))
        elif len(im.shape) == 3:
            return torch.from_numpy(im.transpose(2, 0, 1))


class Compose(object):
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, im):
        for op in self.ops:
            im = op(im)
        return im