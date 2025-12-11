import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image, ImageOps
import glob
import os
import random
import sys

# --- 1. IMPORT CÁC MODEL (Cố gắng thử mọi cái tên có thể) ---

# Thử import MLP
try:
    from src.models.model import SimpleMLP
except ImportError:
    print("❌ Lỗi: Không tìm thấy class SimpleMLP")
    sys.exit(1)

# Thử import CNN (Thử tên 'CNN' trước, nếu không có thì thử 'SimpleCNN')
TargetCNNClass = None
HAS_CNN = False

try:
    from src.models.model import CNN
    TargetCNNClass = CNN
    HAS_CNN = True
    print("✅ Đã tìm thấy class: CNN")
except ImportError:
    try:
        from src.models.model import SimpleCNN
        TargetCNNClass = SimpleCNN
        HAS_CNN = True
        print("✅ Đã tìm thấy class: SimpleCNN")
    except ImportError:
        print("⚠️ Cảnh báo: Không tìm thấy class CNN hay SimpleCNN. Chỉ chạy được MLP.")
        HAS_CNN = False


# --- 2. HÀM LOAD MODEL ---
def load_model(device, model_type):
    model = None
    path = ""
    
    if model_type == 'mlp':
        print("\n🔄 Đang khởi tạo SimpleMLP...")
        model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
        path = "assets/model_final.pth" 
        
    elif model_type == 'cnn':
        if not HAS_CNN or TargetCNNClass is None:
            print("❌ Code model.py chưa có class CNN!")
            return None
            
        print(f"\n🔄 Đang khởi tạo {TargetCNNClass.__name__}...")
        
        # Thử khởi tạo (có tham số hoặc không tham số)
        try:
            model = TargetCNNClass(num_classes=10).to(device)
        except TypeError:
            model = TargetCNNClass().to(device)
            
        # ⚠️ QUAN TRỌNG: Kiểm tra tên file CNN trong folder assets
        # Nếu ông lưu tên khác (vd: best_model.pth) thì nhớ sửa dòng dưới này!
        path = "assets/model_cnn_final.pth"
    
    # Kiểm tra file tồn tại
    if not os.path.exists(path):
        print(f"\n❌ LỖI: Không tìm thấy file trọng số '{path}'")
        print(f"👉 Bạn chọn model {model_type.upper()} nhưng file .pth không có ở đó.")
        print("👉 Kiểm tra lại xem thằng bạn ông lưu file tên gì?")
        return None

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"✅ Đã load thành công: {path}")
        return model
    except Exception as e:
        print(f"❌ Lỗi khi load file trọng số: {e}")
        return None

# --- 3. CHẾ ĐỘ 1: TEST ẢNH TỰ VẼ ---
def test_custom_images(model, device):
    print("\n--- 🎨 CHẾ ĐỘ TEST ẢNH TỰ VẼ (Folder 'inputs') ---")
    image_paths = glob.glob("inputs/*.*")
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = [f for f in image_paths if os.path.splitext(f)[1].lower() in valid_exts]

    if not image_paths:
        print("⚠️ Không tìm thấy ảnh trong folder 'inputs'!")
        return

    # Transform chuẩn
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print(f"🔎 Tìm thấy {len(image_paths)} ảnh. (Tắt cửa sổ để xem tiếp)")

    for img_path in image_paths:
        try:
            orig_img = Image.open(img_path).convert('L')
            
            # Đảo màu nếu nền trắng
            if orig_img.getpixel((0, 0)) > 128: 
                input_img = ImageOps.invert(orig_img)
                note = "Đã đảo màu"
            else:
                input_img = orig_img
                note = "Giữ nguyên"

            img_tensor = transform(input_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0][pred].item() * 100

            print(f"📸 {os.path.basename(img_path)} -> AI đoán: {pred} ({conf:.1f}%)")
            
            plt.figure(figsize=(5, 6))
            plt.imshow(input_img, cmap='gray')
            plt.title(f"Model: {model.__class__.__name__}\nAI đoán: {pred} ({conf:.1f}%)\n[{note}]", color='blue')
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"Lỗi ảnh {img_path}: {e}")

# --- 4. CHẾ ĐỘ 2: TEST MNIST ---
def test_mnist(model, device):
    print("\n--- 🎲 TEST NGẪU NHIÊN TỪ MNIST ---")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    print("👉 Tắt ảnh để xem tấm tiếp theo. Ctrl+C để thoát.\n")

    while True:
        try:
            idx = random.randint(0, len(dataset) - 1)
            img_tensor, label = dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0][pred].item() * 100
            
            status = "ĐÚNG ✅" if pred == label else "SAI ❌"
            color = 'green' if pred == label else 'red'
            
            print(f"Index [{idx}]: Đoán {pred} ({conf:.1f}%) | Thật {label} -> {status}")
            
            plt.figure(figsize=(4, 5))
            plt.imshow(img_tensor.squeeze(), cmap='gray')
            plt.title(f"Model: {model.__class__.__name__}\nĐoán: {pred} ({conf:.1f}%)\nĐáp án: {label}", color=color)
            plt.axis('off')
            plt.show()

        except KeyboardInterrupt:
            break

# --- 5. MENU CHÍNH ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    while True:
        print("\n" + "="*40)
        print("   🤖 MENU DEMO - v2.0")
        print("="*40)
        print("1. Chạy MLP (Cũ)")
        print("2. Chạy CNN (Mới)")
        print("0. Thoát")
        
        choice = input("👉 Chọn (0-2): ")
        
        if choice == '0': break
            
        model = None
        if choice == '1': model = load_model(device, 'mlp')
        elif choice == '2': model = load_model(device, 'cnn')
        else: continue
            
        if model is None: continue

        while True:
            print(f"\n--- 🧠 Model: {model.__class__.__name__} ---")
            print("1. Test ảnh tự vẽ")
            print("2. Test MNIST")
            print("3. Quay lại")
            
            c = input("👉 Chọn: ")
            if c == '1': test_custom_images(model, device)
            elif c == '2': test_mnist(model, device)
            elif c == '3': break

if __name__ == "__main__":
    main()