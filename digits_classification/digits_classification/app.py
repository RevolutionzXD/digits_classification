import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import os
import pandas as pd

# --- 1. SETUP GIAO DIỆN ---
st.set_page_config(page_title="Nhận diện số viết tay", page_icon="🔢", layout="wide")

st.title("🤖 Demo Nhận Diện Số Viết Tay (MNIST)")
st.markdown("Vẽ một con số vào khung bên dưới và xem AI đoán nhé!")

# --- 2. HÀM LOAD MODEL (Dùng Cache để không phải load lại mỗi lần vẽ) ---
@st.cache_resource
def load_model(model_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Import class
    try:
        from src.models.model import SimpleMLP, CNN # Hoặc SimpleCNN tùy tên ông đặt
    except ImportError:
        st.error("❌ Lỗi: Không tìm thấy file model.py. Kiểm tra lại cấu trúc folder!")
        return None, None

    model = None
    path = ""

    if model_type == "CNN (Mạng Tích Chập)":
        try:
            model = CNN(num_classes=10).to(device)
        except:
            model = CNN().to(device) # Fallback nếu không cần tham số
        # ⚠️ SỬA TÊN FILE Ở ĐÂY CHO ĐÚNG MÁY ÔNG
        path = "assets/model_cnn_final.pth" 
    else:
        model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10).to(device)
        path = "assets/model_final.pth"

    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, device
    else:
        st.error(f"⚠️ Không tìm thấy file trọng số: {path}")
        return None, None

# --- 3. SIDEBAR (CHỌN MODEL) ---
with st.sidebar:
    st.header("⚙️ Cấu hình")
    model_choice = st.radio("Chọn Model:", ["CNN (Mạng Tích Chập)", "MLP (Mạng Đa Lớp)"])
    
    model, device = load_model(model_choice)
    
    st.info("Hướng dẫn: Vẽ số to, rõ ở giữa khung hình để AI đoán chuẩn nhất.")

# --- 4. GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Vẽ số ở đây:")
    # Tạo bảng vẽ
    canvas_result = st_canvas(
        fill_color="black",  # Màu nền
        stroke_width=20,     # Nét bút to
        stroke_color="white",# Màu bút trắng
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("2. Kết quả dự đoán:")

    if canvas_result.image_data is not None and model is not None:
        # Lấy dữ liệu ảnh từ bảng vẽ
        img_data = canvas_result.image_data.astype("uint8")
        
        # Chuyển sang ảnh PIL và Grayscale (Đen trắng)
        img = Image.fromarray(img_data).convert("L")
        
        # Resize về 28x28 (Chuẩn MNIST)
        img_resized = img.resize((28, 28))
        
        # Hiển thị cái ảnh mà AI thực sự nhìn thấy (Pixelated)
        st.image(img_resized, width=100, caption="AI nhìn thấy thế này (28x28px)")

        # Nút dự đoán
        if st.button('Dự đoán ngay!'):
            # Preprocessing
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            input_tensor = transform(img_resized).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probs = F.softmax(output, dim=1)[0] * 100 # Chuyển sang %
                
                # Lấy kết quả cao nhất
                pred_label = probs.argmax().item()
                confidence = probs[pred_label].item()

            # --- HIỂN THỊ KẾT QUẢ ---
            st.success(f"🤖 AI đoán là số: **{pred_label}** (Độ tin cậy: {confidence:.1f}%)")
            
            # Vẽ biểu đồ cột (Bar Chart) giống cái ảnh ông gửi
            # Tạo DataFrame cho đẹp
            probs_np = probs.cpu().numpy()
            chart_data = pd.DataFrame(
                probs_np,
                index=[str(i) for i in range(10)],
                columns=["Xác suất (%)"]
            )
            
            st.bar_chart(chart_data)

    elif model is None:
        st.warning("Đang chờ load model...")