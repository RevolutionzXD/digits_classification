import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

def get_dataloaders(data_dir='data', batch_size=64):
    """
    Hàm tải dữ liệu MNIST, chia thành train/val/test và trả về các DataLoader.
    """
    
    # 1. Định nghĩa các phép biến đổi ảnh (Transform)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 2. Tải dữ liệu (Dataset)
    # Sử dụng biến 'data_dir' để khớp với trainer.py
    train_original = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_original = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    # 3. Gộp và Chia dữ liệu theo tỉ lệ 6:2:2
    full_dataset = ConcatDataset([train_original, test_original])
    total_size = len(full_dataset) # 70.000 ảnh
    
    train_size = int(0.6 * total_size) # 42.000 ảnh
    val_size = int(0.2 * total_size)   # 14.000 ảnh
    test_size = total_size - train_size - val_size # 14.000 ảnh

    # Chia ngẫu nhiên, dùng seed=42 để kết quả không đổi
    train_set, val_set, test_set = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 4. Tạo DataLoader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print(f"Data loaded successfully from '{data_dir}'")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    # QUAN TRỌNG: Trả về 3 loader để trainer.py sử dụng
    return train_loader, val_loader, test_loader

# Phần test nhanh (chỉ chạy khi gọi trực tiếp file này)
if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders()
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")