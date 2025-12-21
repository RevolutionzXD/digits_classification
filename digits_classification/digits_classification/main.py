import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image, ImageOps
from src.UI.MainScreenController import UserController
import glob
import os
import random
import sys

def main():
    UserController()

if __name__ == "__main__":
    main()