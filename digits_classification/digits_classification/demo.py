import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import glob
import os
import random
import sys

# --- 1. Cấu trúc Model (Copy y chang lúc train) ---
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Các hàm xử lý ---

def load_model(device):
    model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
    path = "assets/model_final.pth"
    if not os.path.exists(path):
        print(f"❌ Lỗi: Không tìm thấy file '{path}'")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        print("✅ Đã load Model thành công!")
        return model
    except Exception as e:
        print(f"❌ Lỗi file model: {e}")
        sys.exit(1)

def mode_1_custom_images(model, device):
    print("\n--- CHẾ ĐỘ 1: TEST ẢNH TỰ VẼ (FOLDER 'inputs') ---")
    image_paths = glob.glob("inputs/*.*")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [f for f in image_paths if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_paths:
        print("⚠️ Không tìm thấy ảnh nào trong thư mục 'inputs'!")
        return

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"🔎 Tìm thấy {len(image_paths)} ảnh. (Tắt cửa sổ để xem ảnh kế tiếp)")
    
    for img_path in image_paths:
        try:
            # Xử lý ảnh
            orig_img = Image.open(img_path).convert('L')
            if orig_img.getpixel((0, 0)) > 128: # Nếu nền trắng -> Đảo màu
                input_img = ImageOps.invert(orig_img)
                note = "Đã đảo màu nền"
            else:
                input_img = orig_img
                note = "Giữ nguyên màu"

            # Dự đoán
            img_tensor = transform(input_img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item() * 100

            # Hiển thị
            print(f"📸 Ảnh: {os.path.basename(img_path)} -> AI đoán: {pred} ({conf:.1f}%)")
            
            plt.figure(figsize=(4, 5))
            plt.imshow(input_img, cmap='gray')
            plt.title(f"AI đoán: {pred}\n({conf:.1f}%)\n[{note}]", color='blue', fontsize=14)
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Lỗi ảnh {img_path}: {e}")

def mode_2_mnist_random(model, device):
    print("\n--- CHẾ ĐỘ 2: TEST NGẪU NHIÊN TỪ MNIST ---")
    print("⏳ Đang tải dữ liệu Test...")
    
    # Transform hiển thị (chỉ ToTensor)
    tf_display = transforms.ToTensor()
    # Transform dự đoán (thêm Normalize)
    tf_predict = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset_display = datasets.MNIST(root='data', train=False, download=True, transform=tf_display)
    dataset_predict = datasets.MNIST(root='data', train=False, download=True, transform=tf_predict)
    
    print("👉 Tắt cửa sổ ảnh để xem tấm tiếp theo. Bấm Ctrl+C trong terminal để quay lại menu.")

    while True:
        try:
            idx = random.randint(0, len(dataset_display) - 1)
            img_show, label = dataset_display[idx]
            img_in, _ = dataset_predict[idx]

            # Dự đoán
            img_in = img_in.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_in)
                probs = torch.nn.functional.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                conf = probs[0][pred].item() * 100

            status = "ĐÚNG ✅" if pred == label else "SAI ❌"
            color = 'green' if pred == label else 'red'

            print(f"Index [{idx}]: AI đoán {pred} ({conf:.1f}%) | Đáp án: {label} -> {status}")

            plt.figure(figsize=(4, 5))
            plt.imshow(img_show.squeeze(), cmap='gray')
            plt.title(f"AI đoán: {pred} ({conf:.1f}%)\nĐáp án: {label}", color=color, fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.show()
            
        except KeyboardInterrupt:
            break

# --- 3. Chương trình chính ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Khởi động Demo trên: {device}")
    
    model = load_model(device)

    while True:
        print("\n" + "="*30)
        print("   MENU DEMO NHẬN DIỆN SỐ")
        print("="*30)
        print("1. Test ảnh tự vẽ (trong folder 'inputs')")
        print("2. Test ngẫu nhiên từ tập MNIST")
        print("0. Thoát")
        
        choice = input("👉 Chọn chế độ (0-2): ")

        if choice == '1':
            mode_1_custom_images(model, device)
        elif choice == '2':
            mode_2_mnist_random(model, device)
        elif choice == '0':
            print("👋 Tạm biệt!")
            break
        else:
            print("❌ Chọn sai rồi, nhập lại đi ông!")

if __name__ == "__main__":
    main()