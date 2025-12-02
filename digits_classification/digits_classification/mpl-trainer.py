import torch
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm # Thanh tiến trình cho đẹp
import os

# Import các module do nhóm tự viết
from src.models.model import SimpleMLP
from src.losses.loss import get_loss_function
from src.data.dataloader import get_dataloaders

def train_model():
    # 1. Load cấu hình từ file yaml
    config = OmegaConf.load('configs/config.yaml')
    
    # Thiết lập thiết bị (GPU nếu có, không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 2. Chuẩn bị dữ liệu
    print("Đang tải dữ liệu...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data.data_dir, 
        batch_size=config.train.batch_size
    )

    # 3. Khởi tạo Mô hình, Loss, Optimizer
    model = SimpleMLP(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim
    ).to(device)

    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)

    # 4. Vòng lặp Huấn luyện (Training Loop)
    num_epochs = config.train.epoch
    
    for epoch in range(num_epochs):
        model.train() # Chuyển sang chế độ train
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Thanh tiến trình (Progress Bar)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            # --- Các bước huấn luyện cơ bản ---
            optimizer.zero_grad()           # Xóa gradient cũ
            outputs = model(images)         # Lan truyền xuôi (Forward)
            loss = criterion(outputs, labels) # Tính lỗi (Loss)
            loss.backward()                 # Lan truyền ngược (Backward)
            optimizer.step()                # Cập nhật trọng số

            # Tính toán thống kê
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Cập nhật thanh tiến trình
            progress_bar.set_postfix(loss=running_loss/len(train_loader), acc=100.*correct/total)

        # 5. Đánh giá (Validation) sau mỗi epoch
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")

    # 6. Lưu mô hình sau khi huấn luyện xong
    if not os.path.exists("assets"):
        os.makedirs("assets")
    torch.save(model.state_dict(), "assets/model_final.pth")
    print("Đã lưu mô hình vào assets/model_final.pth")

def evaluate(model, loader, device):
    model.eval() # Chuyển sang chế độ đánh giá (không học nữa)
    correct = 0
    total = 0
    with torch.no_grad(): # Không tính gradient để tiết kiệm bộ nhớ
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train_model()