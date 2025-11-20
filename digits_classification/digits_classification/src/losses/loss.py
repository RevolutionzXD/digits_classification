import torch.nn as nn

def get_loss_function():
    # CrossEntropyLoss là tiêu chuẩn cho bài toán phân loại (Classification)
    return nn.CrossEntropyLoss()