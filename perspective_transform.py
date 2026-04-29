import cv2
import numpy as np

def order_points(pts):
    """
    对点进行排序：左上、右上、右下、左下
    :param pts: 4个点 (4x2)
    :return: 排序后的点
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def calculate_document_size(corners):
    """
    计算文档的目标尺寸
    :param corners: 排序后的四个角点 (左上、右上、右下、左下)
    :return: (宽度, 高度)
    """
    (tl, tr, br, bl) = corners
    
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    max_height = max(int(height_left), int(height_right))
    
    if max_width < 10:
        max_width = 10
    if max_height < 10:
        max_height = 10
    
    return max_width, max_height

def get_transformation_matrix(src_points, dst_points):
    """
    获取透视变换矩阵
    """
    return cv2.getPerspectiveTransform(src_points, dst_points)

def apply_perspective_warp(image, M, output_size):
    """
    应用透视变换
    :param image: 输入图像
    :param M: 变换矩阵
    :param output_size: (宽度, 高度)
    :return: 变换后的图像
    """
    return cv2.warpPerspective(
        image, M, output_size,
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

def correct_perspective(image, corners):
    """
    执行完整的透视校正
    :param image: 原始图像
    :param corners: 四个角点
    :return: (校正后的图像, 变换矩阵, 排序后的角点, 目标角点)
    """
    ordered_corners = order_points(corners)
    
    width, height = calculate_document_size(ordered_corners)
    
    dst_corners = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)
    
    M = get_transformation_matrix(ordered_corners, dst_corners)
    
    warped = apply_perspective_warp(image, M, (width, height))
    
    return warped, M, ordered_corners, dst_corners

def is_corners_valid(corners, image_shape):
    """
    检查角点是否有效
    """
    if corners is None:
        return False
    
    if len(corners) != 4:
        return False
    
    h, w = image_shape[:2]
    
    for pt in corners:
        x, y = pt
        if x < -100 or x > w + 100 or y < -100 or y > h + 100:
            return False
    
    return True
