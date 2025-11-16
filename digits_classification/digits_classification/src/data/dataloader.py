import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dataloaders(batch_size: int = 64):
    
    train_original = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_original = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_original_size = len(train_original)
    train_size = int(train_original_size * 0.8)  
    val_size = train_original_size - train_size             
    
    train_dataset, val_dataset = random_split(
        train_original, 
        [train_size, val_size],
        
        generator=torch.Generator().manual_seed(42)
    )
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    test_loader = DataLoader(test_original, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print("--- Phan chia du lieu hoan chinh ---")
    print(f" - Tap Huan Luyen (Train): {len(train_dataset)} mau")
    print(f" - Tap Kiem Dinh (Validation): {len(val_dataset)} mau")
    print(f" - Tap Danh Gia Cuoi Cung (Test): {len(test_original)} mau") 
    print("-----------------------------------")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)