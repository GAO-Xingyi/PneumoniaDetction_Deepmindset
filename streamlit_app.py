import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
import os
import numpy as np

# 页面标题
st.title("Pneumonia Detection Image Classifier")
st.write("Upload a Chest X-ray image OR provide an image URL")

# 类别标签映射
idx_to_class = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral_Pneumonia'}

# 图像预处理方法（与训练保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载模型
@st.cache_resource
def load_model():
    model = models.vgg16_bn(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Linear(25088, 198),
        nn.ReLU(),
        nn.Linear(198, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 4)
    )
    model_path = os.path.join(os.path.dirname(__file__), 'best_model_stage2.pt')
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# 推理函数
def diagnose_chest_xray(img):
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    predicted_idx = np.argmax(probs)
    predicted_class = idx_to_class[predicted_idx]
    confidence_score = probs[predicted_idx]

    if confidence_score >= 0.80:
        confidence_category = 'Very Confident'
    elif 0.65 <= confidence_score < 0.80:
        confidence_category = 'Fairly Confident'
    else:
        confidence_category = 'Potentially Misclassified'

    return predicted_class, confidence_score, confidence_category

# 上传图片 or 输入 URL
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
image_url = st.text_input("Or enter an image URL...")

image = None

# 从本地上传读取图像
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {e}")

# 或从 URL 加载图像
elif image_url:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(image_url, headers=headers)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Error fetching image: {e}")

# 显示图像并执行预测
if image is not None:
    st.image(image, caption="Chest X-ray", use_container_width=True)
    label, prob, confidence = diagnose_chest_xray(image)
    st.success(f"**Predicted Class:** {label}")
    st.info(f"**Confidence Score:** {prob:.2f} — {confidence}")
