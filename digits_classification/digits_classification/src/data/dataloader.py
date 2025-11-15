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

    full_dataset = ConcatDataset([train_original, test_original])
    full_size = len(full_dataset) 

    train_ratio = 0.6
    val_ratio = 0.2
    
    train_size = int(full_size * train_ratio) 
    val_size = int(full_size * val_ratio)   
    test_size = full_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f" DataLoaders da duoc tao theo ty le 6:2:2:")
    print(f" - Tap Huan Luyen (60%): {len(train_dataset)} mau")
    print(f" - Tap Kiem Dinh (20%): {len(val_dataset)} mau")
    print(f" - Tap Kiem Thu (20%): {len(test_dataset)} mau")
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)