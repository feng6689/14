import cv2
import numpy as np

def order_corners(corners):
    """
    将四个角点按左上、右上、右下、左下排序
    :param corners: 四个角点 (4x2 numpy array)
    :return: 排序后的角点
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    
    return rect

def calculate_target_size(corners):
    """
    计算目标矩形尺寸
    :param corners: 排序后的四个角点
    :return: (宽度, 高度)
    """
    (tl, tr, br, bl) = corners
    
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    max_height = max(int(height_left), int(height_right))
    
    return max_width, max_height

def get_perspective_transform(src_points, dst_points):
    """
    获取透视变换矩阵
    :param src_points: 源点（文档角点）
    :param dst_points: 目标点
    :return: 变换矩阵
    """
    return cv2.getPerspectiveTransform(src_points, dst_points)

def apply_perspective_transform(image, M, output_size):
    """
    应用透视变换
    :param image: 输入图像
    :param M: 变换矩阵
    :param output_size: 输出尺寸 (宽, 高)
    :return: 变换后的图像
    """
    return cv2.warpPerspective(image, M, output_size, flags=cv2.INTER_CUBIC)

def transform_document(image, corners):
    """
    完整的文档透视变换流程
    :param image: 原始图像
    :param corners: 检测到的四个角点
    :return: (变换后的图像, 变换矩阵, 原始角点, 目标角点)
    """
    ordered_corners = order_corners(corners)
    
    width, height = calculate_target_size(ordered_corners)
    
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = get_perspective_transform(ordered_corners, dst_corners)
    
    warped = apply_perspective_transform(image, M, (width, height))
    
    return warped, M, ordered_corners, dst_corners
