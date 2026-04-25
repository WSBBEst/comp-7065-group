# Outfit Match - 智能穿搭推荐系统演示文档

## 📋 项目概述

Outfit Match 是一个基于深度学习的智能穿搭推荐系统，能够自动识别用户上传的服装图片，分析搭配协调性，并提供专业的时尚建议。

### 核心功能

- **全身图智能分析**：上传1-5张全身穿搭照片，AI自动识别衣物
- **自动检测与分割**：使用 Mask R-CNN 模型检测和抠图
- **智能搭配推荐**：基于训练好的兼容性预测模型评分
- **专业时尚建议**：提供色彩、风格改进建议和潮流标签

---

## 🏗️ 系统架构

### 技术栈

**前端**
- HTML5 + CSS3（深色主题，响应式设计）
- Vanilla JavaScript（无框架依赖）
- 拖拽上传、实时预览

**后端**
- Python 3.x + Flask
- PyTorch（深度学习框架）
- 双模型协同：
  - **Vision Model**: Mask R-CNN（衣物检测和分割）
  - **Fashion Model**: ResNet + Mean/LSTM（兼容性预测）

**数据存储**
- 文件系统存储（按类别分类：top/pants/shoes等）
- 临时文件自动清理机制



## 🔄 业务流程

### 主要流程：全身图分析（推荐）

```
用户操作流程：
1. 访问 http://localhost:5000
2. 拖拽或点击上传 1-5 张全身穿搭照片
3. 点击"开始智能分析"
4. 等待 AI 处理（10-30秒）
5. 查看分析结果和搭配建议

系统处理流程：
┌─────────────────────────────────────────────┐
│ 1. 前端接收用户上传的图片                      │
│    - 验证文件格式（JPG/PNG/WEBP）             │
│    - 验证文件大小（<16MB）                    │
│    - 限制数量（1-5张）                        │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 2. 发送到后端 API                             │
│    POST /api/analyze-fullbody                │
│    Content-Type: multipart/form-data         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 3. Vision 模型处理（Mask R-CNN）             │
│    - 加载 vision/last_ckpt.pth              │
│    - 对每张图片进行检测                       │
│    - 识别衣物类别（上衣/裤子/鞋子等）          │
│    - 生成分割掩码（mask）                    │
│    - 抠图并去除背景                           │
│    - 白色背景裁剪                             │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 4. 分类保存                                  │
│    - 上衣 → uploads/top/                    │
│    - 裤子 → uploads/pants/                  │
│    - 鞋子 → uploads/shoes/                  │
│    - 连衣裙 → uploads/dress/                │
│    - 命名格式：{category}_{idx}_{timestamp}.png │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 5. Fashion 模型评分（CP Model）              │
│    - 加载 models/cp_best_seed43.pth         │
│    - 从每个类别选择一件单品                   │
│    - 构建搭配组合（top + pants + shoes）     │
│    - 调用 score_outfit_paths() 计算兼容性    │
│    - 输出 0-1 之间的兼容性分数               │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 6. 生成评价和建议                             │
│    - 根据分数生成整体评价                     │
│    - 色彩搭配建议                            │
│    - 风格匹配建议                            │
│    - 潮流标签（3-6个）                       │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 7. 返回 JSON 响应                            │
│    {                                         │
│      "success": true,                        │
│      "detected_items": {...},                │
│      "combination": {...},                   │
│      "compatibility_score": 0.85,            │
│      "evaluation": "...",                    │
│      "color_suggestion": "...",              │
│      "style_suggestion": "...",              │
│      "trend_tags": ["...", "..."]            │
│    }                                         │
└──────────────┬──────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────┐
│ 8. 前端展示结果                              │
│    - 显示识别到的所有衣物                     │
│    - 展示推荐搭配组合图片                     │
│    - 圆形进度条显示评分（0-10分）             │
│    - 文字评价和建议                          │
│    - 风格标签云                              │
└─────────────────────────────────────────────┘
```

### 备选流程1：分类上传

```
用户操作：
1. 分别上传上衣、裤子、鞋子图片
2. 系统自动抠图并保存
3. 直接进行搭配评分

API 端点：POST /api/recommend
适用场景：已有清晰的单品图片
```

### 备选流程2：多图直接分析

```
用户操作：
1. 上传2+张已抠图的单品图片
2. 不进行分割处理
3. 直接评分

API 端点：POST /api/upload
适用场景：快速测试，无需抠图
```

---

## 📁 目录结构

```
comp-7065-group/
├── frontend/                    # 前端文件
│   ├── index.html              # 主页面
│   ├── css/
│   │   └── style.css           # 样式文件（深色主题）
│   └── js/
│       └── main.js             # 前端逻辑
│
├── backend/                     # 后端文件
│   ├── app.py                  # Flask 应用入口
│   ├── config.py               # 配置文件
│   ├── model_inference.py      # 模型推理核心
│   ├── requirements.txt        # Python 依赖
│   ├── vision/                 # Vision 模型
│   │   ├── last_ckpt.pth      # Mask R-CNN 权重（~XXX MB）
│   │   └── load model and weight.py
│   ├── models/                 # Fashion 模型
│   │   └── cp_best_seed43.pth # CP 模型权重（96 MB）
│   └── uploads/                # 上传文件存储
│       ├── top/               # 上衣图片
│       ├── pants/             # 裤子图片
│       ├── shoes/             # 鞋子图片
│       ├── dress/             # 连衣裙
│       └── outwear/           # 外套
│
├── polyvore_route_a.py         # 核心模型实现
├── closet_recommend.ipynb      # 衣橱推荐 Notebook
└── vision_preprocess_to_closet.ipynb  # 视觉预处理 Notebook
```

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.1.1+
- Flask 3.0.0+
- 至少 4GB RAM（CPU 模式）

### 安装步骤

```bash
# 1. 进入后端目录
cd backend

# 2. 安装依赖
pip install flask==3.0.0 flask-cors==4.0.0 pillow==10.1.0 \
    torch==2.1.1 torchvision==0.16.1 numpy==1.26.2

# 3. 确认模型文件存在
# - backend/vision/last_ckpt.pth
# - backend/models/cp_best_seed43.pth

# 4. 启动服务
python app.py

# 5. 访问应用
# 浏览器打开：http://localhost:5000
```

### 验证安装

```bash
# 检查后端健康状态
curl http://localhost:5000/api/health

# 预期返回：
# {"status": "ok", "message": "Outfit Match API is running"}
```

---

## 🎯 使用示例

### 示例1：单张全身照分析

**输入：**
- 1张全身穿搭照片（包含上衣+裤子+鞋子）

**处理：**
- Vision 模型检测出3件衣物
- 自动分类保存
- CP 模型评分：0.82

**输出：**
```json
{
  "success": true,
  "detected_items": {
    "top": [{"label": "short_sleeve_top", "confidence": 0.92}],
    "pants": [{"label": "trousers", "confidence": 0.88}],
    "shoes": [{"label": "sneakers", "confidence": 0.85}]
  },
  "combination": {
    "top": "/uploads/top/top_0_20260421_230000.png",
    "pants": "/uploads/pants/pants_0_20260421_230001.png",
    "shoes": "/uploads/shoes/shoes_0_20260421_230002.png"
  },
  "compatibility_score": 0.82,
  "evaluation": "这是一套不错的搭配！上衣、裤子和鞋子的组合整体协调性较好...",
  "color_suggestion": "色彩搭配基本和谐，可以尝试让三个单品中的两个采用相近色系...",
  "style_suggestion": "整体风格方向正确，建议注意版型的协调性...",
  "trend_tags": ["舒适休闲", "轻商务风", "配饰点睛", "文艺清新"]
}
```

### 示例2：多张全身照组合

**输入：**
- 3张不同穿搭的全身照

**处理：**
- 共检测出 8 件衣物（3上衣 + 3裤子 + 2鞋子）
- 系统选择最佳组合进行评分
- 最终评分：0.91

**优势：**
- 更多选择，找到最佳搭配
- 可以混搭不同照片中的单品

---

## 🔧 API 接口文档

### 1. 健康检查

```http
GET /api/health
```

**响应：**
```json
{
  "status": "ok",
  "message": "Outfit Match API is running"
}
```

---

### 2. 全身图分析（主要接口）

```http
POST /api/analyze-fullbody
Content-Type: multipart/form-data
```

**请求参数：**
- `images`: 文件数组（1-5个图片文件）

**响应：**
```json
{
  "success": true,
  "detected_items": {
    "top": [
      {
        "image": "/uploads/top/top_0_xxx.png",
        "label": "short_sleeve_top",
        "confidence": 0.92,
        "source_image": 1,
        "auto_detected": true
      }
    ],
    "pants": [...],
    "shoes": [...]
  },
  "combination": {
    "top": "/uploads/top/top_0_xxx.png",
    "pants": "/uploads/pants/pants_1_xxx.png",
    "shoes": "/uploads/shoes/shoes_0_xxx.png"
  },
  "compatibility_score": 0.85,
  "overall_evaluation": "这套搭配非常出色！...",
  "color_suggestion": "色彩搭配已经非常完美...",
  "style_suggestion": "风格搭配得当且富有层次感...",
  "trend_tags": ["高级感穿搭", "极简主义", "同色系搭配"],
  "mode": "real",
  "vision_available": true
}
```

**错误响应：**
```json
{
  "success": false,
  "error": "最多只能上传5张图片"
}
```

---

### 3. 分类上传推荐

```http
POST /api/recommend
Content-Type: multipart/form-data
```

**请求参数：**
- `top`: 上衣图片
- `pants`: 裤子图片
- `shoes`: 鞋子图片

**响应：** 同全身图分析

---

### 4. 直接多图分析

```http
POST /api/upload
Content-Type: multipart/form-data
```

**请求参数：**
- `images`: 文件数组（至少2个）

**说明：** 不进行分割处理，直接评分

---

### 5. 静态资源

```http
GET /uploads/{category}/{filename}
```

**类别：** top, pants, shoes, dress, outwear, skirt

---

## 📊 评分系统

### 兼容性分数（Compatibility Score）

- **范围**：0.0 - 1.0
- **含义**：衣物搭配的协调程度
- **计算**：通过训练好的 CP 模型预测

### 综合评分（Overall Score）

- **范围**：0.0 - 10.0
- **转换**：`compatibility_score × 10`
- **等级划分**：
  - 8.5 - 10.0：极佳搭配
  - 7.0 - 8.4：良好搭配
  - 5.0 - 6.9：一般搭配
  - 0.0 - 4.9：需要改进

### 维度评分

- **色彩搭配（Color Match）**：颜色和谐度
- **风格匹配（Style Match）**：风格一致性
- **潮流指数（Trend Score）**：时尚趋势符合度

---

## 🎨 前端界面

### 主要组件

1. **上传区域**
   - 拖拽上传支持
   - 实时预览网格
   - 删除按钮
   - 数量指示器（x/5）

2. **结果展示**
   - 识别衣物网格
   - 推荐搭配组合（横向排列）
   - 圆形评分进度条
   - 详细评价文本
   - 建议卡片（色彩+风格）
   - 标签云

3. **状态提示**
   - 加载动画
   - Vision 模型状态警告
   - 错误消息（3秒自动消失）

### 响应式设计

- **桌面端**：左右分栏布局
- **平板**：上下堆叠
- **手机**：单列布局，优化触摸体验

---

## ⚙️ 配置说明

### backend/config.py

```python
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    
    MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models')
    DEFAULT_MODEL = 'cp_best_seed43.pth'
```

### 修改上传限制

```python
# 在 app.py 中修改
if len(files) > 5:  # 改为其他数字
    return jsonify({'success': False, 'error': '最多只能上传X张图片'}), 400
```

### 修改前端限制

```javascript
// 在 main.js 中修改
const MAX_FILES = 5;  // 改为其他数字
```

---

## 🐛 常见问题

### Q1: Vision 模型未加载

**现象：** 提示"自动检测功能未启用"

**原因：** `backend/vision/last_ckpt.pth` 文件不存在

**解决：**
1. 获取预训练的 Mask R-CNN 权重文件
2. 放置到 `backend/vision/last_ckpt.pth`
3. 重启后端服务

---

### Q2: 模型加载缓慢

**原因：** CPU 模式下推理较慢

**解决：**
- 首次加载需要 5-10 秒（正常）
- 后续请求会复用模型实例（快速）
- 如有 GPU，可修改 `self.device = torch.device("cuda")`

---

### Q3: 上传失败

**检查清单：**
- [ ] 文件格式是否正确（JPG/PNG/WEBP）
- [ ] 文件大小是否 < 16MB
- [ ] 后端服务是否运行
- [ ] CORS 是否正常（跨域时）

---

### Q4: 评分偏低

**可能原因：**
- 图片质量差，检测不准确
- 衣物风格差异大
- 颜色搭配不协调

**建议：**
- 上传清晰的全身照
- 确保光线充足
- 尝试上传多张照片提供更多选择

---

