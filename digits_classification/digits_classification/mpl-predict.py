# predict.py
import torch
from src.models.model import SimpleMLP
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang sử dụng thiết bị: {device}")

# Load model
input_dim = 28*28  # 784
hidden_dim = 128    # thay theo config model của bạn
output_dim = 10
model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
model.load_state_dict(torch.load("assets/model_final.pth", map_location=device))
model.eval()

# Chọn kiểu nhập dữ liệu
pint("Chọn kiểu nhập dữ liệu:")
print("1: Nhập từ ảnh 28x28 (PNG/JPG)")
print("2: Nhập mảng 784 số từ bàn phím")
choice = input("Lựa chọn (1 hoặc 2): ")

if choice == "1":
    # Nhập từ ảnh
    img_path = input("Nhập đường dẫn ảnh: ")
    img_path = img_path.strip().strip('"')
    img = Image.open(img_path).convert("L")  # grayscale
    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Resize giữ tỉ lệ + padding
    from PIL import ImageOps
    img.thumbnail((20,20))  # chữ số vừa trong ô 20x20
    img = ImageOps.pad(img, (28,28), color=0)  # padding background=0 (đen)

    # Nếu nền trắng -> invert màu
    img = ImageOps.invert(img)

    img_tensor = transform(img).unsqueeze(0).to(device)
    img_tensor = transform(img).view(1, -1).to(device)  # flatten + batch dimension

elif choice == "2":
    # Nhập mảng 784 số
    data = input("Nhập 784 số cách nhau bằng space: ")
    nums = [float(x) for x in data.split()]
    if len(nums) != 784:
        raise ValueError("Bạn phải nhập đúng 784 số!")
    img_tensor = torch.tensor(nums, dtype=torch.float32).view(1, -1).to(device)

else:
    raise ValueError("Lựa chọn không hợp lệ!")

#Dự đoán
with torch.no_grad():
    output = model(img_tensor)
    predicted = torch.argmax(output, dim=1).item()

print("Dự đoán chữ số:", predicted)
