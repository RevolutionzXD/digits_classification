from PIL import Image, ImageOps
from torchvision import transforms
import torch


mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def ChangePilToMnistTensor(pilImg: Image.Image, device):

    pilImg = pilImg.convert("L")

    meanValue = sum(pilImg.getdata()) / (pilImg.size[0] * pilImg.size[1])
    if meanValue > 128:
        pilImg = ImageOps.invert(pilImg)

    tensor = mnist_transform(pilImg)
    return tensor.unsqueeze(0).to(device)  # (1,1,28,28)
