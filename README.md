## COMP7065 Group Project

本仓库包含 Polyvore “Route A” + 视觉抠图预处理 + 衣橱穿搭推荐的实现与演示，覆盖：

- Compatibility Prediction（CP，兼容性二分类）
- Fill-In-The-Blank（FITB，多选题评估/训练）
- Vision 预处理（检测 + 分割 + 抠图；支持 Notebook 内上传图片）
- Closet Recommendation（从本地衣橱图片组合并打分，输出 Top-K 穿搭）

核心实现：[`polyvore_route_a.py`](file:///d:/Projects/JupyterProject/COMP7065_GROUP/polyvore_route_a.py)  
演示 Notebook：[`polyvore_route_a_rewrite.ipynb`](file:///d:/Projects/JupyterProject/COMP7065_GROUP/polyvore_route_a_rewrite.ipynb)

## 目录结构（关键部分）

- `polyvore_route_a.py`：数据读取 / 模型 / 训练 / 评估 / 推荐 CLI
- `polyvore_route_a_rewrite.ipynb`：Notebook（建议 Run All，从上到下执行）
- `sample.ipynb`：Route A 的最小化演示（stats / train 等）
- `archive/Re-PolyVore/Re-PolyVore/Re-PolyVore/`：图片根目录（按类别/slot 分文件夹）
- `polyvore-dataset-master/polyvore.tar.gz`：Polyvore 标注与 json（CP/FITB 需要）
- `models/`：训练得到的 checkpoint（示例：`cp_best_seed43.pth` 等）
- `vision_preprocess_to_closet.ipynb`：用户拍照预处理（检测 + 分割 + 抠图 + 按 slot 落盘）
- `closet_recommend.ipynb`：加载 checkpoint，在衣橱内采样组合并可视化 Top-K
- `vision/last_ckpt.pth`：视觉模型权重（Mask R-CNN）
- `vision/load model and weight.py`：视觉模型结构定义与权重加载
- `user_upload_raw/`：用户上传/拍照原图目录（没有也可，Notebook 会 fallback 到 demo）
- `user_closet_demo/`：演示用衣橱图片（按 slot 分文件夹）
- `user_closet_processed/`：抠图/清洗后的衣橱目录（vision 预处理输出）
- `vision_test/`：用于推荐/测试的示例图片

## 环境准备

本项目主要依赖：

- Python 3.x
- `torch`, `torchvision`
- `Pillow`
- Notebook 可选：`jupyter`
- `numpy`（vision 预处理用）
- 画图可选：`matplotlib`
- 上传 UI 可选：`ipywidgets`（vision 预处理里提供上传控件；没有也能走本地目录输入）

`requirements.txt` 当前为空，如果要统一环境，建议在组内约定一个固定的安装方式（例如 conda 环境或 pip freeze 后补齐）。

## 数据说明

需要两个输入：

1. 图片目录 `--images-root`  
   默认示例在本仓库：`archive/Re-PolyVore/Re-PolyVore/Re-PolyVore/`

2. Polyvore 标注 tar `--polyvore-tar`  
   默认路径：`polyvore-dataset-master/polyvore.tar.gz`

说明：不同数据包里 tar 内部文件名可能有差异（例如 `fashion_compatibility_prediction.txt` vs `fashion-compatibility-prediction.txt`），`polyvore_route_a.py` 已做兼容解析。

## 快速开始（Notebook）

打开 [`polyvore_route_a_rewrite.ipynb`](file:///d:/Projects/JupyterProject/COMP7065_GROUP/polyvore_route_a_rewrite.ipynb)，直接选择 “Run All”。

- Notebook 会自动定位 `project_root`、`images_root`、`polyvore_tar`
- 训练参数（例如 epoch 数）可在对应 cell 里调整
- 会在 `models/` 下写出 checkpoint（由 args 决定）

## 从拍照到穿搭（抠图 → 衣橱 → 推荐）

推荐的组内演示流程：

1) 打开 [`vision_preprocess_to_closet.ipynb`](file:///d:/Projects/JupyterProject/COMP7065_GROUP/vision_preprocess_to_closet.ipynb) 并 Run All  
   - 加载 `vision/last_ckpt.pth`（Mask R-CNN）  
   - 输入方式：
     - 直接往 `user_upload_raw/` 放图片；或
     - 在 Notebook 内用上传控件上传（需要 `ipywidgets`）  
   - 输出：抠图后的图片会按 slot 写入 `user_closet_processed/`（文件名带 `_raw`/`_clean`）

2) 打开 [`closet_recommend.ipynb`](file:///d:/Projects/JupyterProject/COMP7065_GROUP/closet_recommend.ipynb) 并 Run All  
   - 自动优先使用 `user_closet_processed/`，否则 fallback 到 `user_closet_demo/`
   - 加载 `models/` 下的 CP checkpoint，对衣橱里不同 slot 进行组合并输出 Top-K

## 快速开始（命令行）

在仓库根目录执行（Windows PowerShell 示例）：

### 1) 统计可用样本

```bash
python polyvore_route_a.py stats --images-root "archive\Re-PolyVore\Re-PolyVore\Re-PolyVore"
```

### 2) 训练 CP（二分类）

```bash
python polyvore_route_a.py train-cp --images-root "archive\Re-PolyVore\Re-PolyVore\Re-PolyVore" --epochs 3 --save-path "models\cp.pth"
```

### 3) 训练 FITB（多选）

```bash
python polyvore_route_a.py train-fitb --images-root "archive\Re-PolyVore\Re-PolyVore\Re-PolyVore" --epochs 3 --save-path "models\fitb.pth"
```

### 4) 评估 FITB（使用 checkpoint）

```bash
python polyvore_route_a.py eval-fitb --images-root "archive\Re-PolyVore\Re-PolyVore\Re-PolyVore" --checkpoint "models\fitb.pth"
```

### 5) Closet 推荐（从本地衣橱图片组合打分）

```bash
python polyvore_route_a.py recommend-closet --closet-root "vision_test" --checkpoint "models\cp.pth" --slots top pants shoes --top-k 10
```

可选参数：

- `--cpu`：强制 CPU
- `--backbone resnet18|resnet50`：模型 backbone
- `--arch mean|bilstm`：聚合方式（mean 或双向 LSTM）

## 组内协作建议（GitHub）

- 大文件（图片、tar、ckpt）不建议直接推到 GitHub 仓库；建议用 Git LFS 或共享网盘/Release，并在组内约定下载放置路径。
- 提交前确认：Notebook 输出不要包含过大的二进制/图片（可清理 outputs）。
