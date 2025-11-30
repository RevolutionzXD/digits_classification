import torch
import torch.optim as optim
from omegaconf import OmegaConf
from tqdm import tqdm 
import os

from src.models.model import CNN
from src.losses.loss import get_loss_function
from src.data.dataloader import get_dataloaders

def train_cnn():
    
    config = OmegaConf.load('configs/config.yaml')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    print("Đang tải dữ liệu...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=config.data.data_dir, 
        batch_size=config.train.batch_size
    )

    print("Đang khởi tạo model CNN...")
    model = CNN(output_dim=config.model_cnn.output_dim).to(device)

    criterion = get_loss_function()
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)

    num_epochs = config.train.epoch
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()          
            outputs = model(images)         
            loss = criterion(outputs, labels) 
            loss.backward()                
            optimizer.step()               

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=running_loss/len(train_loader), acc=100.*correct/total)

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {100.*correct/total:.2f}% | Val Acc: {val_acc:.2f}%")

    if not os.path.exists("assets"):
        os.makedirs("assets")
    torch.save(model.state_dict(), "assets/model_cnn_final.pth")
    print("Đã lưu mô hình CNN vào assets/model_cnn_final.pth")

def evaluate(model, loader, device):
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == "__main__":
    train_cnn()