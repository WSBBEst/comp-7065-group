import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from config import Config
from model_inference import get_model_inference, process_and_recommend_outfit, process_fullbody_images

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    # 支持多张图片上传
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': 'No image files provided'}), 400

    files = request.files.getlist('images')
    
    # 过滤空文件
    files = [f for f in files if f.filename != '']
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': 'No selected files'}), 400

    if len(files) < 2:
        return jsonify({'success': False, 'error': '请至少上传2张图片进行穿搭分析'}), 400

    saved_files = []
    try:
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = str(hash(file.filename + str(os.urandom(8))))
                timestamp_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], timestamp_filename)
                file.save(filepath)
                saved_files.append(filepath)
            else:
                return jsonify({'success': False, 'error': f'Invalid file type: {file.filename}'}), 400

        # 调用模型分析多张图片
        result = get_model_inference(saved_files)
        
        # 清理临时文件
        for filepath in saved_files:
            if os.path.exists(filepath):
                os.remove(filepath)
                
        return jsonify(result)
        
    except Exception as e:
        # 发生错误时清理所有临时文件
        for filepath in saved_files:
            if os.path.exists(filepath):
                os.remove(filepath)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/recommend', methods=['POST'])
def recommend_outfit():
    """
    接收分类上传的图片，自动抠图后保存到对应目录，返回最佳搭配推荐
    完整流程：上传 → 抠图 → 保存 → 推荐
    """
    required_categories = ['top', 'pants', 'shoes']
    
    # 检查是否所有必需类别都已上传
    for category in required_categories:
        if category not in request.files:
            return jsonify({'success': False, 'error': f'缺少{category}类别的图片'}), 400
    
    try:
        # 构建分类文件字典
        category_files = {}
        for category in required_categories:
            file = request.files[category]
            if file and file.filename and allowed_file(file.filename):
                category_files[category] = file
            else:
                return jsonify({'success': False, 'error': f'{category}类别的文件无效'}), 400
        
        # 调用完整的处理流程：抠图 → 保存 → 推荐
        result = process_and_recommend_outfit(category_files, app.config['UPLOAD_FOLDER'])
        
        if result.get('success'):
            # 构建返回结果
            response_data = {
                'success': True,
                'combination': result.get('processed_images', {}),
                'score': result.get('compatibility_score', 0),
                'evaluation': result.get('overall_evaluation', ''),
                'color_suggestion': result.get('color_suggestion', ''),
                'style_suggestion': result.get('style_suggestion', ''),
                'trend_tags': result.get('trend_tags', []),
                'compatibility_score': result.get('compatibility_score', 0),
                'segmentation_info': result.get('segmentation_info', {}),
                'mode': result.get('mode', 'unknown')
            }
            return jsonify(response_data)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"[API] Recommend outfit failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze-fullbody', methods=['POST'])
def analyze_fullbody():
    """
    接收多张全身图，自动检测、分割、分类，然后推荐最佳搭配
    流程：上传全身图 → 检测分割 → 分类保存 → 组合推荐
    """
    if 'images' not in request.files:
        return jsonify({'success': False, 'error': '请至少上传一张全身图'}), 400
    
    files = request.files.getlist('images')
    files = [f for f in files if f.filename != '']
    
    if len(files) == 0:
        return jsonify({'success': False, 'error': '没有有效的图片文件'}), 400
    
    if len(files) > 5:
        return jsonify({'success': False, 'error': '最多只能上传5张图片'}), 400
    
    try:
        # 调用全身图处理函数
        result = process_fullbody_images(files, app.config['UPLOAD_FOLDER'])
        
        if result.get('success'):
            response_data = {
                'success': True,
                'detected_items': result.get('detected_items', {}),
                'combination': result.get('combination', {}),
                'score': result.get('compatibility_score', 0),
                'evaluation': result.get('overall_evaluation', ''),
                'color_suggestion': result.get('color_suggestion', ''),
                'style_suggestion': result.get('style_suggestion', ''),
                'trend_tags': result.get('trend_tags', []),
                'compatibility_score': result.get('compatibility_score', 0),
                'mode': result.get('mode', 'unknown')
            }
            return jsonify(response_data)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"[API] Analyze fullbody failed: {str(e)}")
        logger.exception("[API] Exception details:")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'Outfit Match API is running'})

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory('../frontend/css', filename)

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory('../frontend/js', filename)

@app.route('/uploads/<category>/<filename>')
def serve_uploaded_file(category, filename):
    """提供上传文件的访问"""
    if category not in ['top', 'pants', 'shoes', 'dress', 'outwear', 'skirt']:
        return jsonify({'error': 'Invalid category'}), 404
    
    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], category)
    return send_from_directory(upload_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)