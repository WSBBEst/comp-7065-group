import os
import random
import logging
import shutil
from typing import Dict, Any, Optional, Sequence, List
import torch
from torchvision import transforms
from PIL import Image
import sys
import io
import numpy as np
import math
from datetime import datetime
from torchvision.transforms import functional as F

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path to import polyvore_route_a
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from polyvore_route_a import OutfitCompatModel, load_checkpoint_model, sort_outfit_paths, build_default_transform, score_outfit_paths

# Vision 模型相关 - 修正路径为 backend/vision
VISION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vision')
VISION_CKPT_PATH = os.path.join(VISION_DIR, 'last_ckpt.pth')

logger.info(f"[VISION] Vision directory: {VISION_DIR}")
logger.info(f"[VISION] Checkpoint path: {VISION_CKPT_PATH}")
logger.info(f"[VISION] Checkpoint exists: {os.path.exists(VISION_CKPT_PATH)}")

# DeepFashion2 类别映射（与 vision_preprocess_to_closet.ipynb 保持一致）
CLASS_ID_TO_NAME = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}

CLASS_NAME_TO_SLOT = {
    "short_sleeve_top": "top",
    "long_sleeve_top": "top",
    "vest": "top",
    "sling": "top",
    "short_sleeve_outwear": "outwear",
    "long_sleeve_outwear": "outwear",
    "shorts": "pants",
    "trousers": "pants",
    "skirt": "skirt",
    "short_sleeve_dress": "dress",
    "long_sleeve_dress": "dress",
    "vest_dress": "dress",
    "sling_dress": "dress",
}

class VisionSegmentationModel:
    """视觉分割模型，用于抠图"""
    
    def __init__(self, num_classes=14, device="cpu"):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model = None
        self._load_vision_model()
    
    def _load_vision_model(self):
        """加载 Mask R-CNN 模型"""
        if not os.path.exists(VISION_CKPT_PATH):
            logger.warning(f"[VISION] Model checkpoint not found: {VISION_CKPT_PATH}")
            return
        
        try:
            logger.info(f"[VISION] Loading segmentation model from {VISION_CKPT_PATH}")
            
            # 动态导入 vision loader
            vision_loader_path = os.path.join(VISION_DIR, 'load model and weight.py')
            spec = __import__('importlib.util').util.spec_from_file_location("vision_loader", vision_loader_path)
            vision_loader = __import__('importlib.util').util.module_from_spec(spec)
            spec.loader.exec_module(vision_loader)
            
            self.model = vision_loader.load_model_weights(
                VISION_CKPT_PATH, 
                num_classes=self.num_classes, 
                device=str(self.device)
            )
            logger.info("[VISION] Segmentation model loaded successfully")
        except Exception as e:
            logger.error(f"[VISION] Failed to load segmentation model: {e}")
            self.model = None
    
    def segment_and_crop(self, image: Image.Image, score_threshold=0.55, mask_threshold=0.50, padding_ratio=0.06):
        """对图片进行分割并抠图"""
        if self.model is None:
            logger.warning("[VISION] Model not available, returning original image")
            return {"ok": False, "reason": "model_not_loaded", "img": image, "clean": image}
        
        try:
            # 转换为 tensor
            x = F.to_tensor(image).to(self.device)
            
            with torch.no_grad():
                output = self.model([x])[0]
            
            # 选择最佳实例
            idx = self._choose_main_instance(output, score_threshold)
            if idx is None:
                return {"ok": False, "reason": "no_detection", "img": image, "clean": image}
            
            # 获取类别信息
            label_id = int(output["labels"][idx].detach().cpu().item())
            score = float(output["scores"][idx].detach().cpu().item())
            name = CLASS_ID_TO_NAME.get(label_id, f"class_{label_id}")
            slot = CLASS_NAME_TO_SLOT.get(name, "unknown")
            
            # 获取 mask 并抠图
            mask = output["masks"][idx, 0].detach().cpu().numpy() >= mask_threshold
            clean = self._mask_to_cropped_white_bg(image, mask, padding_ratio)
            
            return {
                "ok": True,
                "img": image,
                "clean": clean,
                "label_id": label_id,
                "label_name": name,
                "slot": slot,
                "score": score,
            }
        except Exception as e:
            logger.error(f"[VISION] Segmentation failed: {e}")
            return {"ok": False, "reason": str(e), "img": image, "clean": image}
    
    def _choose_main_instance(self, output, score_threshold):
        """选择主要的检测实例"""
        scores = output["scores"].detach().cpu()
        boxes = output["boxes"].detach().cpu()
        
        if scores.numel() == 0:
            return None
        
        valid = torch.where(scores >= score_threshold)[0]
        if valid.numel() == 0:
            valid = torch.tensor([int(torch.argmax(scores).item())])
        
        # 兼顾置信度和面积
        best_idx = None
        best_value = -1.0
        for i in valid.tolist():
            x1, y1, x2, y2 = boxes[i].tolist()
            area = max(1.0, (x2 - x1) * (y2 - y1))
            value = float(scores[i]) * math.sqrt(area)
            if value > best_value:
                best_value = value
                best_idx = i
        
        return int(best_idx) if best_idx is not None else None
    
    def _mask_to_cropped_white_bg(self, img, mask_2d, padding_ratio):
        """根据 mask 抠图并裁剪，白色背景"""
        arr = np.array(img)
        h, w = arr.shape[:2]
        
        ys, xs = np.where(mask_2d)
        if len(xs) == 0 or len(ys) == 0:
            return img.copy()
        
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        pad_x = int((x2 - x1 + 1) * padding_ratio)
        pad_y = int((y2 - y1 + 1) * padding_ratio)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w - 1, x2 + pad_x)
        y2 = min(h - 1, y2 + pad_y)
        
        white = np.full_like(arr, 255)
        white[mask_2d] = arr[mask_2d]
        crop = white[y1:y2+1, x1:x2+1]
        return Image.fromarray(crop)


class FashionModelInterface:
    def __init__(self, model_path: str, model_name: str = 'cp_best_seed43.pth'):
        self.model_path = model_path
        self.model_name = model_name
        self.model = None
        self.transform = None
        # 强制使用 CPU 避免 CUDA 兼容性问题
        self.device = torch.device("cpu")
        logger.info(f"[DEVICE] Force using device: {self.device} (CPU mode to avoid CUDA compatibility issues)")
        
        # 初始化视觉分割模型
        self.vision_model = VisionSegmentationModel(device=str(self.device))
        
        self._load_model()

    def _load_model(self):
        logger.info(f"[MODEL] Initializing model: {self.model_name}")
        logger.info(f"[MODEL] Model path: {self.model_path}")
        logger.info(f"[DEVICE] Using device: {self.device}")

        if os.path.exists(self.model_path):
            logger.info(f"[MODEL] Model file found, attempting to load...")
            try:
                self.model, ckpt_args = load_checkpoint_model(self.model_path, self.device)
                image_size = int(ckpt_args.get("image_size", 224))
                self.transform = build_default_transform(image_size)
                logger.info(f"[MODEL] Model loaded successfully!")
                logger.info(f"[MODEL] Model architecture: {ckpt_args.get('arch', 'mean')}")
                logger.info(f"[MODEL] Embed dim: {ckpt_args.get('embed_dim', 256)}")
                logger.info(f"[MODEL] Image size: {image_size}")
                logger.info(f"[MODEL] Backbone: {ckpt_args.get('backbone', 'resnet18')}")
                logger.info(f"[MODEL] Mode: REAL (using trained model)")
            except Exception as e:
                logger.error(f"[MODEL] Error loading model: {str(e)}")
                logger.warning(f"[MODEL] Falling back to mock mode")
                self.model = None
                self.transform = None
        else:
            logger.warning(f"[MODEL] Model path {self.model_path} not found, using mock mode")
            logger.info(f"[MODEL] Mock mode will generate random predictions")

    def process_and_recommend(self, category_images: Dict[str, Any], upload_folder: str) -> Dict[str, Any]:
        """
        处理用户上传的原始图片：抠图 → 保存 → 推荐
        :param category_images: {'top': file_obj, 'pants': file_obj, 'shoes': file_obj}
        :param upload_folder: 上传文件夹路径
        :return: 推荐结果
        """
        logger.info("[PROCESS] Starting process_and_recommend workflow")
        
        saved_paths = {}
        processed_paths = {}
        
        try:
            # Step 1: 对每个类别的图片进行抠图处理
            for category, file_obj in category_images.items():
                logger.info(f"[PROCESS] Processing category: {category}")
                
                # 读取图片
                img = Image.open(file_obj.stream).convert("RGB")
                
                # 抠图
                result = self.vision_model.segment_and_crop(img)
                
                if result["ok"]:
                    logger.info(f"[PROCESS] {category}: Detected as {result.get('label_name', 'unknown')} (slot={result.get('slot', 'unknown')})")
                    processed_img = result["clean"]
                else:
                    logger.warning(f"[PROCESS] {category}: Segmentation failed ({result.get('reason')}), using original image")
                    processed_img = img
                
                # 保存到对应目录
                category_dir = os.path.join(upload_folder, category)
                os.makedirs(category_dir, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{category}_{timestamp}.png"
                filepath = os.path.join(category_dir, filename)
                
                processed_img.save(filepath)
                saved_paths[category] = filepath
                processed_paths[category] = processed_img
                
                logger.info(f"[PROCESS] Saved {category} to {filepath}")
            
            # Step 2: 构建图片路径列表（按顺序：top, pants, shoes）
            required_categories = ['top', 'pants', 'shoes']
            image_paths = [saved_paths[cat] for cat in required_categories if cat in saved_paths]
            
            if len(image_paths) < 3:
                missing = [cat for cat in required_categories if cat not in saved_paths]
                raise ValueError(f"Missing categories: {missing}")
            
            # Step 3: 调用模型进行搭配推荐
            logger.info("[PROCESS] Calling model for outfit recommendation")
            result = self.predict(image_paths)
            
            # Step 4: 添加处理后的图片信息到结果中
            if result.get('success'):
                result['processed_images'] = {
                    cat: f'/uploads/{cat}/{os.path.basename(path)}'
                    for cat, path in saved_paths.items()
                }
                result['segmentation_info'] = {
                    cat: 'success' if cat in saved_paths else 'failed'
                    for cat in required_categories
                }
            
            logger.info("[PROCESS] Process and recommend completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"[PROCESS] Process and recommend failed: {e}")
            logger.exception("[PROCESS] Exception details:")
            
            # 清理临时文件
            for filepath in list(saved_paths.values()):
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
            
            raise

    def predict(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        logger.info(f"[PREDICT] Received prediction request with {len(image_paths)} images")

        for i, image_path in enumerate(image_paths):
            logger.info(f"[PREDICT] Image {i+1}: {image_path}")
            if not os.path.exists(image_path):
                logger.error(f"[PREDICT] Image not found: {image_path}")
                raise FileNotFoundError(f"Image not found: {image_path}")

        if len(image_paths) < 2:
            logger.warning(f"[PREDICT] Less than 2 images provided, returning low score")
            return self._generate_result(0.3, image_paths)

        if self.model and self.transform:
            logger.info(f"[PREDICT] Using real model inference (Closet Recommendation logic)")
            return self._real_inference(image_paths)
        else:
            logger.info(f"[PREDICT] Using mock inference")
            return self._mock_inference(image_paths)

    def _real_inference(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        logger.info(f"[INFERENCE] Starting real inference using score_outfit_paths...")
        start_time = datetime.now()

        try:
            # 使用 polyvore_route_a 中的 score_outfit_paths 函数
            # 这个函数会自动排序图片路径并进行模型推理
            compatibility_score = score_outfit_paths(
                model=self.model,
                transform=self.transform,
                device=self.device,
                outfit_paths=image_paths
            )
            
            end_time = datetime.now()
            inference_time = (end_time - start_time).total_seconds()
            
            logger.info(f"[INFERENCE] Compatibility score (0-1): {compatibility_score:.4f}")
            logger.info(f"[INFERENCE] Inference completed in {inference_time:.3f} seconds")
            
            return self._generate_result(compatibility_score, image_paths)
            
        except Exception as e:
            logger.error(f"[INFERENCE] Real inference failed: {str(e)}")
            logger.exception("[INFERENCE] Exception details:")
            logger.warning(f"[INFERENCE] Falling back to mock inference")
            return self._mock_inference(image_paths)

    def _generate_result(self, compatibility_score: float, image_paths: Sequence[str]) -> Dict[str, Any]:
        """根据兼容性分数生成完整的评价结果"""
        # 将 0-1 的分数转换为 0-10 的综合评分
        overall_score = round(compatibility_score * 10, 1)
        logger.info(f"[RESULT] Overall score (0-10): {overall_score}")

        # 基于兼容性分数生成各个维度的分数
        scores = {
            'overall': overall_score,
            'color_match': round(max(5.0, min(10.0, compatibility_score * 10 * random.uniform(0.92, 1.08))), 1),
            'style_match': round(max(5.0, min(10.0, compatibility_score * 10 * random.uniform(0.90, 1.10))), 1),
            'trend_score': round(max(5.0, min(10.0, compatibility_score * 10 * random.uniform(0.88, 1.12))), 1)
        }
        logger.info(f"[RESULT] Generated scores: {scores}")

        # 根据分数生成评价文本
        if compatibility_score >= 0.85:
            evaluation_text = "This is an excellent outfit combination! The colors of the top, pants, and shoes are harmonious and unified, with consistent styling that demonstrates outstanding fashion taste and matching skills. The overall look aligns with current trends while maintaining personal character."
            color_suggestion = "The color coordination is already perfect, with natural tone transitions across all three pieces. To elevate it further, consider adding accessories (such as a belt, watch, or bag) that echo the main color palette to enhance the overall cohesion."
            style_suggestion = "The style matching is well-balanced and layered. The fit coordination between the top and pants is reasonable, and the shoe style complements the overall aesthetic. Keep this matching approach and consider adding more personal touches in the details."
            trend_tags = ['Premium Style', 'Minimalism', 'Monochrome Coordination', 'Layered Look', 'Quality Choice', 'Fashion Forward']
        elif compatibility_score >= 0.70:
            evaluation_text = "This is a nice outfit combination! The coordination between the top, pants, and shoes shows good overall harmony, with relatively balanced colors and a certain sense of fashion. A few minor adjustments can make the overall effect even more outstanding."
            color_suggestion = "The color coordination is basically harmonious. Try having two of the three pieces in similar color tones, with the third serving as an accent color. For example: if the top and pants are neutral colors, use the shoes as a highlight."
            style_suggestion = "The overall style direction is correct. Pay attention to fit coordination. If the top is loose-fitting, consider pairing it with slim-fit pants for contrast; ensure the shoe style maintains consistency with the overall look to avoid style conflicts."
            trend_tags = ['Casual Comfort', 'Smart Casual', 'Accessory Highlight', 'Fresh & Artistic', 'Versatile Daily']
        elif compatibility_score >= 0.50:
            evaluation_text = "This outfit has a basic foundation, but there's room for improvement in color and style coordination. The combination of top, pants, and shoes can be more thoughtful to achieve a better visual effect."
            color_suggestion = "Consider reducing the number of colors and choosing 2 main tones for coordination. Follow the 'three-color rule': no more than three main colors in the outfit. Try matching the shoes with either the top or pants for better harmony."
            style_suggestion = "Start by defining a clear style direction (such as casual, business, or sporty), then select pieces around this theme. Pay attention to fit coordination, avoiding mismatched combinations of overly loose or tight items."
            trend_tags = ['Basic Match', 'Casual Style', 'Practical', 'Minimalist']
        else:
            evaluation_text = "The coordination of this outfit needs improvement. There are significant differences in color and style between the top, pants, and shoes. Consider revising your piece selection and adjusting according to basic outfit principles."
            color_suggestion = "Start practicing with basic black, white, and gray combinations, then gradually try similar color schemes. Avoid too many bright colors appearing simultaneously. Begin by matching two pieces in similar tones, with the third as a neutral transition."
            style_suggestion = "First determine the occasion and style positioning, then choose pieces accordingly. For example, for work: shirt + trousers + leather shoes; for casual: T-shirt + jeans + sneakers."
            trend_tags = ['Needs Improvement', 'Basic Learning', 'Keep It Simple']

        logger.info(f"[RESULT] Generated evaluation: {evaluation_text}")
        logger.info(f"[RESULT] Color suggestion: {color_suggestion}")
        logger.info(f"[RESULT] Style suggestion: {style_suggestion}")
        logger.info(f"[RESULT] Trend tags: {trend_tags}")

        result = {
            'success': True,
            'scores': scores,
            'overall_evaluation': evaluation_text,
            'color_suggestion': color_suggestion,
            'style_suggestion': style_suggestion,
            'trend_tags': trend_tags,
            'model_version': self.model_name,
            'compatibility_score': round(compatibility_score, 4),
            'mode': 'real' if self.model else 'mock'
        }

        logger.info(f"[RESULT] Returning successful result (mode={result['mode']})")
        return result

    def _mock_inference(self, image_paths: Sequence[str]) -> Dict[str, Any]:
        logger.info(f"[MOCK] Starting mock inference for {len(image_paths)} images")
        logger.warning(f"[MOCK] No real model loaded, using random predictions")

        # 模拟一个合理的兼容性分数
        compatibility_score = random.uniform(0.55, 0.90)
        
        return self._generate_result(compatibility_score, image_paths)

# 全局模型实例（单例模式，避免重复加载）
_model_instance = None

def get_model_inference(image_paths: Sequence[str], model_name: str = 'cp_best_seed43.pth') -> Dict[str, Any]:
    global _model_instance
    
    from config import Config
    logger.info("=" * 60)
    logger.info("[API] get_model_inference called")
    logger.info(f"[API] Model name: {model_name}")
    logger.info(f"[API] Number of images: {len(image_paths)}")
    logger.info(f"[API] Image paths: {list(image_paths)}")

    model_path = os.path.join(Config.MODEL_PATH, model_name)
    logger.info(f"[API] Resolved model path: {model_path}")

    try:
        # 使用单例模式，只在第一次调用时加载模型
        if _model_instance is None:
            logger.info("[API] Creating new model instance (first call)")
            _model_instance = FashionModelInterface(model_path, model_name)
        else:
            logger.info("[API] Reusing existing model instance")
            
        result = _model_instance.predict(image_paths)
        logger.info(f"[API] Prediction successful, success={result.get('success')}, mode={result.get('mode')}")
        logger.info("=" * 60)
        return result
    except Exception as e:
        logger.error(f"[API] Prediction failed: {str(e)}")
        logger.exception("[API] Exception details:")
        logger.info("=" * 60)
        # 发生错误时重置单例，下次重新加载
        _model_instance = None
        raise


def process_and_recommend_outfit(category_files: Dict[str, Any], upload_folder: str, model_name: str = 'cp_best_seed43.pth') -> Dict[str, Any]:
    """
    完整的处理流程：接收分类文件 → 抠图 → 保存 → 推荐
    :param category_files: {'top': file_obj, 'pants': file_obj, 'shoes': file_obj}
    :param upload_folder: 上传文件夹路径
    :param model_name: 模型名称
    :return: 推荐结果
    """
    global _model_instance
    
    from config import Config
    logger.info("=" * 60)
    logger.info("[API] process_and_recommend_outfit called")
    logger.info(f"[API] Categories: {list(category_files.keys())}")
    
    model_path = os.path.join(Config.MODEL_PATH, model_name)
    logger.info(f"[API] Resolved model path: {model_path}")
    
    try:
        # 使用单例模式
        if _model_instance is None:
            logger.info("[API] Creating new model instance (first call)")
            _model_instance = FashionModelInterface(model_path, model_name)
        else:
            logger.info("[API] Reusing existing model instance")
        
        # 调用处理并推荐方法
        result = _model_instance.process_and_recommend(category_files, upload_folder)
        
        logger.info(f"[API] Process and recommend successful, success={result.get('success')}")
        logger.info("=" * 60)
        return result
        
    except Exception as e:
        logger.error(f"[API] Process and recommend failed: {str(e)}")
        logger.exception("[API] Exception details:")
        logger.info("=" * 60)
        # 发生错误时重置单例
        _model_instance = None
        raise


def process_fullbody_images(image_files, upload_folder: str, model_name: str = 'cp_best_seed43.pth') -> Dict[str, Any]:
    """
    处理多张全身图：检测 → 分割 → 分类 → 组合推荐
    :param image_files: Flask 文件对象列表
    :param upload_folder: 上传文件夹路径
    :param model_name: 模型名称
    :return: 推荐结果
    """
    global _model_instance
    
    from config import Config
    logger.info("=" * 60)
    logger.info("[API] process_fullbody_images called")
    logger.info(f"[API] Number of images: {len(image_files)}")
    
    model_path = os.path.join(Config.MODEL_PATH, model_name)
    logger.info(f"[API] Resolved model path: {model_path}")
    
    saved_items = {}  # {category: [filepath1, filepath2, ...]}
    detected_items = {}  # 用于返回前端显示
    
    try:
        # Step 1: 初始化模型
        if _model_instance is None:
            logger.info("[API] Creating new model instance (first call)")
            _model_instance = FashionModelInterface(model_path, model_name)
        else:
            logger.info("[API] Reusing existing model instance")
        
        vision_model = _model_instance.vision_model
        
        # Step 2: 处理每张图片
        for idx, file_obj in enumerate(image_files):
            logger.info(f"[PROCESS] Processing image {idx + 1}/{len(image_files)}")
            
            try:
                # 读取图片
                img = Image.open(file_obj.stream).convert("RGB")
                
                # 分割检测
                result = vision_model.segment_and_crop(img)
                
                if result["ok"]:
                    slot = result.get('slot', 'unknown')
                    label_name = result.get('label_name', 'unknown')
                    confidence = result.get('score', 0)
                    processed_img = result["clean"]
                    
                    logger.info(f"[PROCESS] Image {idx + 1}: Detected {label_name} -> slot={slot}")
                    
                    # 保存到对应类别目录
                    if slot in ['top', 'pants', 'shoes', 'dress', 'outwear', 'skirt', 'bag']:
                        category_dir = os.path.join(upload_folder, slot)
                        os.makedirs(category_dir, exist_ok=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{slot}_{idx}_{timestamp}.png"
                        filepath = os.path.join(category_dir, filename)
                        
                        processed_img.save(filepath)
                        
                        # 记录保存的路径
                        if slot not in saved_items:
                            saved_items[slot] = []
                        saved_items[slot].append(filepath)
                        
                        # 记录检测信息用于前端显示
                        if slot not in detected_items:
                            detected_items[slot] = []
                        detected_items[slot].append({
                            'image': f'/uploads/{slot}/{filename}',
                            'label': label_name,
                            'confidence': confidence,
                            'source_image': idx + 1
                        })
                        
                        logger.info(f"[PROCESS] Saved to {filepath}")
                    else:
                        logger.warning(f"[PROCESS] Unknown slot: {slot}, skipping")
                else:
                    logger.warning(f"[PROCESS] Image {idx + 1}: Detection failed ({result.get('reason')})")
                    
            except Exception as e:
                logger.error(f"[PROCESS] Failed to process image {idx + 1}: {e}")
                continue
        
        # Step 3: 检查是否有足够的物品进行搭配
        required_slots = ['top', 'pants', 'shoes']
        available_slots = {slot: paths for slot, paths in saved_items.items() if slot in required_slots}
        
        if not available_slots:
            vision_available = vision_model.model is not None
            return {
                'success': False,
                'error': 'No valid clothing items detected (top, pants, shoes). Please try uploading clearer full-body photos.',
                'vision_available': vision_available
            }
        
        # Step 4: 构建搭配组合（每个类别取第一件）
        combination_paths = {}
        combination_display = {}
        
        for slot in required_slots:
            if slot in available_slots and available_slots[slot]:
                combination_paths[slot] = available_slots[slot][0]
                combination_display[slot] = f'/uploads/{slot}/{os.path.basename(combination_paths[slot])}'
        
        # 如果缺少某些类别，尝试用其他类别补充
        if 'top' not in combination_paths and 'dress' in saved_items:
            combination_paths['top'] = saved_items['dress'][0]
            combination_display['top'] = f'/uploads/dress/{os.path.basename(combination_paths["top"])}'
        
        if 'pants' not in combination_paths:
            if 'skirt' in saved_items:
                combination_paths['pants'] = saved_items['skirt'][0]
                combination_display['bottom'] = f'/uploads/skirt/{os.path.basename(combination_paths["pants"])}'
            elif 'dress' in saved_items:
                combination_paths['pants'] = saved_items['dress'][0]
                combination_display['bottom'] = f'/uploads/dress/{os.path.basename(combination_paths["pants"])}'
        
        # Step 5: 如果有至少2个类别，进行搭配评分
        if len(combination_paths) >= 2:
            image_paths = list(combination_paths.values())
            logger.info(f"[PROCESS] Calling model for recommendation with {len(image_paths)} items")
            result = _model_instance.predict(image_paths)
            
            if result.get('success'):
                result['detected_items'] = detected_items
                result['combination'] = combination_display
                result['available_slots'] = list(available_slots.keys())
                logger.info("[PROCESS] Fullbody analysis completed successfully")
                return result
        else:
            logger.warning("[PROCESS] Not enough items for combination")
        
        # 如果无法组合，返回检测到的物品信息
        vision_available = vision_model.model is not None
        return {
            'success': True,
            'detected_items': detected_items,
            'combination': combination_display,
            'compatibility_score': 0,
            'overall_evaluation': 'The detected clothing items are insufficient to form a complete outfit. Please upload full-body photos containing more pieces.',
            'color_suggestion': '',
            'style_suggestion': '',
            'trend_tags': [],
            'mode': 'real' if _model_instance.model else 'mock',
            'vision_available': vision_available
        }
        
    except Exception as e:
        logger.error(f"[API] Process fullbody images failed: {str(e)}")
        logger.exception("[API] Exception details:")
        
        # 清理临时文件
        for slot, paths in saved_items.items():
            for filepath in paths:
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except:
                        pass
        
        raise
