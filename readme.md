# U²-Net Predict and Show

A simple project to run U²-Net image prediction and display results.

---

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/thatsillegal/u2net_predict_and_show.git
cd u2net_predict_and_show
```

### 2. 创建 Conda 环境并初始化依赖
使用 environment.yml 创建环境：
```bash
conda env create -f environment.yml
```
激活环境：
```bash
conda activate u2net_py37
```
### 3. 运行项目
```bash
python app.py
```
默认端口 5000，可以通过浏览器访问：
```bash
http://localhost:5000
```
上传图片并查看预测结果。
## 目录结构
```bash
/saved_models/u2net/   # U²-Net 模型文件
/static/               # 前端静态资源
/uploads/              # 上传图片
app.py                 # Flask 主程序
environment.yml        # Conda 环境依赖
```
### 说明

- /saved_models/u2net/：存放 U²-Net 模型文件。
- /static/：前端静态资源，如 CSS、JS、示例图片。
- /uploads/：用户上传的图片。
- app.py：Flask 主程序，提供上传和预测接口。
- environment.yml：Conda 环境依赖文件，用于快速初始化环境。