import cv2
import numpy as np
import json
import os
import sys

from preprocess import preprocess_image_multi_strategy, enhance_edges_strong
from boundary_detection import (
    find_document_with_multiple_strategies,
    order_points
)
from perspective_transform import (
    correct_perspective,
    is_corners_valid
)
from enhancement import enhance_document, get_enhancement_methods


def load_image(image_path):
    """
    加载图像
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件: {image_path}")
        return None
    
    return image


def save_json(corners, transform_matrix, output_path):
    """
    保存角点和变换矩阵到JSON
    """
    ordered = order_points(corners)
    
    json_data = {
        "corners": {
            "top_left": [float(ordered[0][0]), float(ordered[0][1])],
            "top_right": [float(ordered[1][0]), float(ordered[1][1])],
            "bottom_right": [float(ordered[2][0]), float(ordered[2][1])],
            "bottom_left": [float(ordered[3][0]), float(ordered[3][1])]
        },
        "transform_matrix": transform_matrix.tolist() if transform_matrix is not None else None
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"已保存: {output_path}")


def get_default_corners(image_shape):
    """
    获取默认角点（图像四个顶点）
    """
    h, w = image_shape[:2]
    return np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)


def print_corners(corners):
    """
    打印角点坐标
    """
    ordered = order_points(corners)
    print("四个原始角点像素坐标:")
    print(f"  左上: ({int(ordered[0][0])}, {int(ordered[0][1])})")
    print(f"  右上: ({int(ordered[1][0])}, {int(ordered[1][1])})")
    print(f"  右下: ({int(ordered[2][0])}, {int(ordered[2][1])})")
    print(f"  左下: ({int(ordered[3][0])}, {int(ordered[3][1])})")


def process_document(image_path, output_dir=".", enhancement_method='balanced'):
    """
    主处理流程
    """
    print(f"正在处理图像: {image_path}")
    print("-" * 60)
    
    image = load_image(image_path)
    if image is None:
        print("错误: 无法加载图像")
        return
    
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    print("\n正在进行图像预处理...")
    preprocess_results = preprocess_image_multi_strategy(image)
    
    print("\n正在检测文档边界...")
    corners, is_estimated = find_document_with_multiple_strategies(preprocess_results, image.shape)
    
    use_default_corners = False
    transform_matrix = None
    scanned_image = None
    
    if corners is None or not is_corners_valid(corners, image.shape):
        print("未检测到文档区域，使用原图边界")
        corners = get_default_corners(image.shape)
        use_default_corners = True
    else:
        print(f"检测到文档边界{'（估算）' if is_estimated else ''}")
    
    print_corners(corners)
    
    if not use_default_corners:
        print("\n正在执行透视变换...")
        try:
            scanned_image, transform_matrix, ordered_corners, dst_corners = correct_perspective(
                image, corners
            )
            
            doc_w, doc_h = scanned_image.shape[1], scanned_image.shape[0]
            print(f"文档尺寸: {doc_w} x {doc_h}")
            
            if doc_w < 50 or doc_h < 50:
                print("文档区域过小，使用原图")
                scanned_image = image.copy()
                use_default_corners = True
        except Exception as e:
            print(f"透视变换出错: {e}")
            scanned_image = image.copy()
            use_default_corners = True
    else:
        scanned_image = image.copy()
    
    print("\n正在增强图像...")
    print(f"增强方法: {get_enhancement_methods().get(enhancement_method, enhancement_method)}")
    
    try:
        enhanced_image = enhance_document(scanned_image, method=enhancement_method)
    except Exception as e:
        print(f"增强出错: {e}，使用灰度替代")
        if len(scanned_image.shape) == 3:
            enhanced_image = cv2.cvtColor(scanned_image, cv2.COLOR_BGR2GRAY)
        else:
            enhanced_image = scanned_image.copy()
    
    print("\n正在保存结果...")
    
    scanned_path = os.path.join(output_dir, "scanned.jpg")
    cv2.imwrite(scanned_path, scanned_image)
    print(f"已保存校正图像: {scanned_path}")
    
    enhanced_path = os.path.join(output_dir, "enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced_image)
    print(f"已保存增强图像: {enhanced_path}")
    
    json_path = os.path.join(output_dir, "1.json")
    save_json(corners, transform_matrix, json_path)
    
    print("-" * 60)
    print("处理完成！")
    print("\n生成的文件:")
    print("  - scanned.jpg: 透视校正后的图像")
    print("  - enhanced.jpg: 增强处理后的图像")
    print("  - 1.json: 角点坐标和变换矩阵")
    
    if use_default_corners:
        print("\n注意: 未检测到有效文档边界，使用了图像边界")


def main():
    """
    主函数
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "1.jpg"
    
    if len(sys.argv) > 2:
        enhancement_method = sys.argv[2]
    else:
        enhancement_method = 'balanced'
    
    valid_methods = ['gray', 'color', 'binary', 'balanced']
    if enhancement_method not in valid_methods:
        print(f"警告: 未知的增强方法 '{enhancement_method}'，使用默认的 'balanced'")
        print(f"可用方法: {', '.join(valid_methods)}")
        enhancement_method = 'balanced'
    
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)
    
    output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
    
    process_document(image_path, output_dir, enhancement_method)


if __name__ == "__main__":
    main()
