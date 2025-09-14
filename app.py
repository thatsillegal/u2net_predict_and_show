import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from get_prediction import predict_mask
from get_contours import process_all_images
import time
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST']) # 定义一个处理post请求的HTTP接口
def process_image(): # 定义一个函数来处理接口的请求
    file = request.files['image'] # 从POST请求中提取名为'image'的文件对象
    timestamp = int(time.time())
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    os.makedirs('uploads', exist_ok=True)
    input_path = os.path.join('uploads',filename) # 为了安全性，使用 Flask 提供的 secure_filename() 处理上传的原始文件名
    file.save(input_path)
    mask_path = predict_mask(input_path)
    contours, res_path = process_all_images(os.path.basename(mask_path))

    # 编码图片为 base64
    with open(res_path, 'rb') as f:
        image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    image_url = f"data:image/png;base64,{image_base64}"

    # 把 numpy 的 contours 转为普通 list
    simplified_contours = []
    for cnt in contours:
        simplified = []
        for pt in cnt:
            x, y = pt[0]  # cnt 是 (n,1,2)
            simplified.append([int(x), int(y)])
        simplified_contours.append(simplified)

    json_obj = {
        'image': image_url,
        'contours': simplified_contours
    }

    return jsonify(json_obj)

if __name__ == '__main__':
    app.run(debug=True)