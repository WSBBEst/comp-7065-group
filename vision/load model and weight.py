import torch
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_trained_model(num_classes):
    """
    最小化模型定义：和你训练时的模型结构完全一致
    :param num_classes: 你的数据集类别数（包含背景）
    """
    # 加载基础MaskRCNN模型
    model = maskrcnn_resnet50_fpn(pretrained=False)  # 不加载预训练权重，只拿结构

    # 替换分类头（修改后的模型结构核心）
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 替换掩码头（修改后的模型结构核心）
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_channels=in_features_mask,
        dim_reduced=hidden_layer,
        num_classes=num_classes
    )

    return model

def load_model_weights(model_path, num_classes, device="cpu"):
    """
    读取你训练好的权重文件
    :param model_path: 权重文件路径（.pth / .pt）
    :param num_classes: 你的模型类别数（必须和训练时一致）
    :param device: cpu / cuda
    :return: 加载好权重的模型
    """
    # 初始化模型结构
    model = get_trained_model(num_classes)
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])  # 标准训练保存格式
    model.to(device)
    model.eval()  # 推理模式
    return model

# ===================== 组员使用示例 =====================
if __name__ == "__main__":
    # 1. 修改为你的模型参数
    NUM_CLASSES = 14  # 替换成你训练时的类别数（DeepFashion2默认13类+背景=14）
    WEIGHT_PATH = "your_trained_weights.pth"  # 权重文件路径

    # 2. 加载模型
    model = load_model_weights(WEIGHT_PATH, NUM_CLASSES)
    print("✅ 模型权重加载成功！")