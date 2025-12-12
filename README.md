# digits_classification
![Biểu đồ Grantt của nhóm](./digits_classification/digits_classification/assets/gian_do_grantt.png)
# Digits Classification - Phân loại chữ số viết tay (MNIST)

Dự án xây dựng mô hình trí tuệ nhân tạo (AI) đơn giản để nhận diện và phân loại chữ số viết tay từ bộ dữ liệu MNIST. Dự án được thực hiện bằng ngôn ngữ Python và thư viện PyTorch, quản lý môi trường bằng Micromamba.

## Tính năng chính
* Mô hình: Mạng nơ-ron đa lớp (MLP - Multilayer Perceptron).
           Mạng tích chập(CNN - Convolutional Neutral Network)
* Dữ liệu: Tự động tải, xử lý và phân chia bộ dữ liệu MNIST.
* Cấu trúc: Mã nguồn được tổ chức dạng module (src/data, src/models, src/losses).
* Cấu hình: Quản lý tham số huấn luyện tập trung qua file config.yaml.
* Kết quả: Độ chính xác (Accuracy) đạt khoảng 97% - 99% sau 10 epochs.

## Cấu trúc dự án

```
digits_classification/
├── assets/               # Chứa file model đã huấn luyện (.pth)
├── configs/
│   └── config.yaml       # File cấu hình tham số
├── data/                 # Thư mục chứa dữ liệu MNIST
├── inputs/               # Thư mục chứa ảnh tự vẽ để test
├── src/
│   ├── data/
│   │   └── dataloader.py
│   ├── losses/
│   │   └── loss.py
│   └── models/
│       └── model.py
├── app.py                # Giao diện Web (Streamlit)
├── demo.py               # Demo CLI (Terminal)
├── trainer.py            # Script huấn luyện MLP 
├── trainer_cnn.py        # Script huấn luyện CNN 
├── requirements.txt
└── README.md
```

## Cài đặt và Môi trường

Dự án yêu cầu Python 3.10 trở lên. Khuyến khích sử dụng Micromamba hoặc Conda để quản lý môi trường.

### 1. Clone dự án
```
git clone <https://github.com/RevolutionzXD/digits_classification>
cd digits_classification
```

### 2. Tạo môi trường ảo
```
micromamba create -n digits_classification python=3.10 -y
micromamba activate digits_classification
```

### 3. Cài đặt thư viện
```
pip install -r requirements.txt
```

## Cấu hình (Configuration)

Bạn có thể thay đổi các tham số trong file configs/config.yaml:

```
train:
  epoch: 10             # Số vòng lặp huấn luyện
  batch_size: 64        # Kích thước lô dữ liệu
  learning_rate: 0.001  # Tốc độ học

data:
  data_dir: "data"      # Tên thư mục lưu dữ liệu

model:
  input_dim: 784        # Đầu vào (28x28 pixel)
  hidden_dim: 128       # Số nơ-ron lớp ẩn
  output_dim: 10        # Đầu ra (10 chữ số)
```

## Hướng dẫn Huấn luyện

Để bắt đầu huấn luyện mô hình, hãy chạy lệnh sau tại thư mục gốc:

```
python3 trainer.py      #train mlp
python3 trainer_cnn.py  #train cnn
```

Quá trình thực hiện:
1. Tự động tải dữ liệu MNIST về thư mục data/ (nếu chưa có).
2. Chia dữ liệu thành 3 tập: Train (60%), Validation (20%), Test (20%).
3. Huấn luyện mô hình qua số lượng Epoch đã định.
4. Lưu mô hình tốt nhất vào thư mục assets/model_final.pth.

## Kết quả Demo

SimpleMLP	
*Accuracy(Validation): ~97.13%
*Đặc điểm: Nhanh, nhẹ, cấu trúc đơn giản.
SimpleCNN		
*Accuracy(Validation): ~99.20%
*Đặc điểm: Chính xác cao hơn, nhận diện tốt các biến thể hình ảnh.

## Thành viên nhóm

* [Nguyễn Thiện Nhân] - Trưởng nhóm & Git Master
* [Phạm Hồng Minh] - Xử lý dữ liệu
* [Đặng Giang Linh] - Xử lí dữ liệu
* [Phạm Thành Lộc] - Xử lí dữ liệu
* [Phạm Duy Thái] - Xây dựng mô hình
* [Lê Anh Minh] - Xây dựng mô hình
* [Mai Kim Khôi] - Xây dựng mô hình


---
Đồ án môn Nhập môn Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM



Markdown# DIGITS_CLASSIFICATION
![Biểu đồ Grantt của nhóm](./digits_classification/digits_classification/assets/gian_do_grantt.png)

# Digits Classification - Phân loại chữ số viết tay (MNIST)

Dự án xây dựng mô hình trí tuệ nhân tạo (AI) để nhận diện và phân loại chữ số viết tay từ bộ dữ liệu MNIST. Dự án được thực hiện bằng **Python** và **PyTorch**, hỗ trợ cả giao diện dòng lệnh (CLI) và giao diện Web trực quan.

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Demo-Streamlit-green?logo=streamlit)

## Tính năng chính
* **Đa mô hình:**
    * Mạng nơ-ron đa lớp (**MLP** - Multilayer Perceptron).
    * Mạng tích chập (**CNN** - Convolutional Neural Network).
* **Demo Web App:** Giao diện đồ họa (Streamlit) cho phép vẽ số trực tiếp lên màn hình.
* **Dữ liệu:** Tự động tải, xử lý và phân chia bộ dữ liệu MNIST.
* **Cấu hình:** Quản lý tham số huấn luyện tập trung qua file `config.yaml`.

##  Cấu trúc dự án

```text
digits_classification/
├── assets/               # Chứa file model đã huấn luyện (.pth)
├── configs/
│   └── config.yaml       # File cấu hình tham số
├── data/                 # Thư mục chứa dữ liệu MNIST (Tự động tải)
├── inputs/               # Thư mục chứa ảnh tự vẽ để test
├── src/
│   ├── data/
│   │   └── dataloader.py # Xử lý dữ liệu
│   ├── losses/
│   │   └── loss.py       # Hàm mất mát
│   └── models/
│       └── model.py      # Kiến trúc mạng (MLP & CNN)
├── app.py                # Giao diện Web (Streamlit)
├── demo.py               # Demo CLI (Terminal)
├── trainer.py            # Script huấn luyện MLP
├── trainer_cnn.py        # Script huấn luyện CNN
├── requirements.txt      # Danh sách thư viện
└── README.md             # Tài liệu hướng dẫn
```

## Cài đặt và Môi trường
Dự án yêu cầu Python 3.10 trở lên. Khuyến khích sử dụng Micromamba hoặc Conda.
1. Clone dự án
```
git clone [https://github.com/RevolutionzXD/digits_classification](https://github.com/RevolutionzXD/digits_classification)
cd digits_classification
```
2. Tạo môi trường ảo
```
Bashmicromamba create -n digits_classification python=3.10 -y
micromamba activate digits_classification
```
3. Cài đặt thư viện
```
pip install -r requirements.txt
```
## Cấu hình (Configuration)
Bạn có thể thay đổi các tham số trong file configs/config.yaml:
  epoch: 10             # Số vòng lặp huấn luyện
  batch_size: 64        # Kích thước lô dữ liệu
  learning_rate: 0.001

model:
  input_dim: 784        # Đầu vào MLP (28x28 pixel)
  hidden_dim: 128       # Số nơ-ron lớp ẩn
  output_dim: 10        # Đầu ra (10 chữ số)
## Hướng dẫn Huấn luyện (Training)
Dự án cung cấp 2 script riêng biệt cho từng loại mô hình:
1. Huấn luyện MLP (Cơ bản)
```
python3 trainer.py
```
Output: assets/model_final.pth
2. Huấn luyện CNN (Nâng cao - Khuyên dùng)
```
python3 trainer_cnn.py
```
Output: assets/model_cnn_final.pth
Lưu ý: Quá trình sẽ tự động tải dữ liệu về thư mục data/ và chia tập Train/Val/Test.
## Hướng dẫn chạy Demo
Sau khi có file model, bạn có 2 cách để kiểm thử:
Cách 1: Chạy Web App (Giao diện)Vẽ số trực tiếp lên màn hình và xem AI dự đoán theo thời gian thực.
```
streamlit run app.py
```
Sau khi chạy, giữ phím Ctrl và click vào link http://localhost:8501 hiện ra trong terminal.
Cách 2: Chạy CLI Demo (Terminal)Menu cổ điển, test với ảnh trong folder inputs hoặc ảnh MNIST ngẫu nhiên.Bashpython3 demo.py


## Kết quả Demo

SimpleMLP	
*Accuracy(Validation): ~97.13%
*Đặc điểm: Nhanh, nhẹ, cấu trúc đơn giản.
SimpleCNN		
*Accuracy(Validation): ~99.20%
*Đặc điểm: Chính xác cao hơn, nhận diện tốt các biến thể hình ảnh.

## Thành viên nhóm

* [Nguyễn Thiện Nhân] - Trưởng nhóm & Git Master
* [Phạm Hồng Minh] - Xử lý dữ liệu
* [Đặng Giang Linh] - Xử lí dữ liệu
* [Phạm Thành Lộc] - Xử lí dữ liệu
* [Phạm Duy Thái] - Xây dựng mô hình
* [Lê Anh Minh] - Xây dựng mô hình
* [Mai Kim Khôi] - Xây dựng mô hình


---
Đồ án môn Nhập môn Công nghệ Thông tin - Trường Đại học Khoa học Tự nhiên, ĐHQG-HCM