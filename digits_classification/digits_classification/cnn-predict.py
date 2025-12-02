import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# --- CNN architecture phải giống trainer ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

# --- Setup device + load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("assets/cnn_model.pth", map_location=device))
model.eval()

# --- Nhập ảnh ---
img_path = input("Nhập đường dẫn ảnh: ").strip().strip('"')
if not os.path.exists(img_path):
    raise FileNotFoundError(f"File {img_path} không tồn tại")

img = Image.open(img_path).convert("L")
img = img.resize((28,28))  # resize về 28x28
# Resize giữ tỉ lệ + padding
from PIL import ImageOps
img.thumbnail((20,20))  # chữ số vừa trong ô 20x20
img = ImageOps.pad(img, (28,28), color=0)  # padding background=0 (đen)

# Nếu nền trắng -> invert màu
img = ImageOps.invert(img)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img).unsqueeze(0).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
img_tensor = transform(img).unsqueeze(0).to(device)  # batch=1

# --- Dự đoán ---
with torch.no_grad():
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1).item()

print("Dự đoán chữ số:", predicted)
