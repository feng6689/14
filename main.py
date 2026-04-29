import cv2
import numpy as np
import json
import os
import sys

from preprocess import preprocess_image, enhance_edges
from boundary_detection import (
    find_largest_quadrilateral,
    estimate_corners_with_hough,
    get_default_corners
)
from perspective_transform import transform_document, order_corners
from enhancement import enhance_document, color_enhance


def load_image(image_path):
    """
    加载图像
    :param image_path: 图像路径
    :return: 图像对象或 None
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误: 无法读取图像文件: {image_path}")
        return None
    
    return image


def save_results(scanned_image, enhanced_image, corners, transform_matrix, output_dir="."):
    """
    保存结果
    :param scanned_image: 校正后的图像
    :param enhanced_image: 增强后的图像
    :param corners: 四个角点坐标
    :param transform_matrix: 变换矩阵
    :param output_dir: 输出目录
    """
    scanned_path = os.path.join(output_dir, "scanned.jpg")
    cv2.imwrite(scanned_path, scanned_image)
    print(f"已保存校正图像: {scanned_path}")
    
    enhanced_path = os.path.join(output_dir, "enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced_image)
    print(f"已保存增强图像: {enhanced_path}")
    
    json_data = {
        "corners": {
            "top_left": [float(corners[0][0]), float(corners[0][1])],
            "top_right": [float(corners[1][0]), float(corners[1][1])],
            "bottom_right": [float(corners[2][0]), float(corners[2][1])],
            "bottom_left": [float(corners[3][0]), float(corners[3][1])]
        },
        "transform_matrix": transform_matrix.tolist()
    }
    
    json_path = os.path.join(output_dir, "1.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4, ensure_ascii=False)
    print(f"已保存角点和变换矩阵: {json_path}")


def save_original_as_result(image, output_dir="."):
    """
    保存原图作为结果（当检测不到文档时）
    :param image: 原始图像
    :param output_dir: 输出目录
    """
    scanned_path = os.path.join(output_dir, "scanned.jpg")
    cv2.imwrite(scanned_path, image)
    print(f"已保存原图作为校正图像: {scanned_path}")
    
    enhanced = enhance_document(image, method='clahe_adaptive')
    enhanced_path = os.path.join(output_dir, "enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced)
    print(f"已保存增强图像: {enhanced_path}")


def process_document(image_path, output_dir="."):
    """
    主处理流程
    :param image_path: 输入图像路径
    :param output_dir: 输出目录
    """
    print(f"正在处理图像: {image_path}")
    print("-" * 50)
    
    image = load_image(image_path)
    if image is None:
        print("错误: 无法加载图像，程序退出")
        return
    
    original_image = image.copy()
    h, w = image.shape[:2]
    print(f"图像尺寸: {w} x {h}")
    
    edges, gray = preprocess_image(image)
    
    corners = find_largest_quadrilateral(edges, min_area_ratio=0.03)
    
    if corners is None:
        print("尝试增强边缘后再次检测...")
        enhanced_edges = enhance_edges(edges)
        corners = find_largest_quadrilateral(enhanced_edges, min_area_ratio=0.02)
        
        if corners is None:
            print("未找到清晰边界，采用估算角点")
            corners = estimate_corners_with_hough(edges, gray)
    
    if corners is None:
        print("未检测到文档区域，保存原图")
        save_original_as_result(original_image, output_dir)
        print("四个原始角点像素坐标（图像边界）:")
        print(f"  左上: (0, 0)")
        print(f"  右上: ({w-1}, 0)")
        print(f"  右下: ({w-1}, {h-1})")
        print(f"  左下: (0, {h-1})")
        return
    
    corners = order_corners(corners)
    
    print("四个原始角点像素坐标:")
    print(f"  左上: ({int(corners[0][0])}, {int(corners[0][1])})")
    print(f"  右上: ({int(corners[1][0])}, {int(corners[1][1])})")
    print(f"  右下: ({int(corners[2][0])}, {int(corners[2][1])})")
    print(f"  左下: ({int(corners[3][0])}, {int(corners[3][1])})")
    
    try:
        scanned_image, transform_matrix, ordered_corners, dst_corners = transform_document(
            original_image, corners
        )
    except Exception as e:
        print(f"透视变换出错: {e}")
        print("未检测到有效文档区域，保存原图")
        save_original_as_result(original_image, output_dir)
        return
    
    if scanned_image.shape[0] < 50 or scanned_image.shape[1] < 50:
        print("检测到的文档区域过小，保存原图")
        save_original_as_result(original_image, output_dir)
        return
    
    try:
        enhanced_image = enhance_document(scanned_image, method='clahe_adaptive')
    except Exception as e:
        print(f"图像增强出错: {e}，使用CLAHE替代")
        enhanced_image = enhance_document(scanned_image, method='clahe')
    
    save_results(scanned_image, enhanced_image, ordered_corners, transform_matrix, output_dir)
    
    print("-" * 50)
    print("处理完成！")
    print("生成的文件:")
    print("  - scanned.jpg: 透视校正后的扫描件")
    print("  - enhanced.jpg: 增强处理后的图像")
    print("  - 1.json: 角点坐标和变换矩阵")


def main():
    """
    主函数
    """
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "1.jpg"
    
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.getcwd(), image_path)
    
    output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
    
    process_document(image_path, output_dir)


if __name__ == "__main__":
    main()
