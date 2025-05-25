from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from flask_cors import CORS

# Flask setup
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
UPLOAD_FOLDER = 'static/uploads/'

# 设置当前目录和上传路径
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
app.config['UPLOAD_FOLDER'] = os.path.join(current_directory, UPLOAD_FOLDER)

# 创建上传文件夹（如不存在）
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 模型加载
model_path = os.path.join(current_directory, 'best_model_stage2.pt')

# 构建与训练时相同结构的 VGG16 模型
model = models.vgg16_bn(pretrained=False)
model.classifier = nn.Sequential(
    nn.Linear(25088, 198),
    nn.ReLU(),
    nn.Linear(198, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 4)
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 类别标签映射
idx_to_class = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral_Pneumonia'}

# 图像预处理（与训练一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 图像加载与预处理
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    return img_tensor

# 推理函数
def diagnose_chest_xray(image_path):
    img_tensor = load_and_preprocess_image(image_path)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).numpy()[0]

    predicted_idx = np.argmax(probs)
    predicted_class = idx_to_class[predicted_idx]
    confidence_score = probs[predicted_idx]

    # 置信度分类
    if confidence_score >= 0.80:
        confidence_category = 'Very Confident'
    elif 0.65 <= confidence_score < 0.80:
        confidence_category = 'Fairly Confident'
    else:
        confidence_category = 'Potentially Misclassified'

    return predicted_class, confidence_category

# 首页
@app.route("/")
def index():
    return render_template("PneumoniaDetction.htm")

# 上传接口
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        label, confidence = diagnose_chest_xray(file_path)

        return jsonify({
            'success': True,
            'filename': 'static/uploads/' + filename,
            'result': label,
            'confidence': confidence
        })

    return jsonify({'success': False, 'message': 'Invalid file type'}), 400

# 文件格式检查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# 启动服务
if __name__ == '__main__':
    app.run(debug=True)
