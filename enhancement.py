import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    应用CLAHE（对比度受限自适应直方图均衡化）
    :param image: 输入图像 (BGR或灰度)
    :param clip_limit: 对比度限制阈值
    :param tile_grid_size: 网格大小
    :return: 增强后的图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    
    return enhanced

def adaptive_threshold_enhance(image, block_size=11, C=2):
    """
    应用自适应阈值增强
    :param image: 输入图像 (BGR或灰度)
    :param block_size: 邻域大小（奇数）
    :param C: 从平均值或加权平均值中减去的常数
    :return: 增强后的二值图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    enhanced = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C
    )
    
    return enhanced

def remove_shadow(image):
    """
    移除阴影，改善不均匀光照
    :param image: 输入图像 (BGR)
    :return: 去除阴影后的灰度图像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    
    background = cv2.medianBlur(dilated, 21)
    
    diff = 255 - cv2.subtract(background, gray)
    
    return diff

def enhance_document(image, method='clahe_adaptive'):
    """
    综合文档增强方法
    :param image: 输入图像 (BGR)
    :param method: 增强方法
        - 'clahe': 仅CLAHE
        - 'adaptive': 仅自适应阈值
        - 'clahe_adaptive': CLAHE + 自适应阈值
        - 'shadow_removal': 阴影移除 + CLAHE
    :return: 增强后的图像
    """
    if method == 'clahe':
        return apply_clahe(image)
    
    elif method == 'adaptive':
        return adaptive_threshold_enhance(image)
    
    elif method == 'clahe_adaptive':
        clahe_enhanced = apply_clahe(image)
        enhanced = adaptive_threshold_enhance(clahe_enhanced)
        return enhanced
    
    elif method == 'shadow_removal':
        shadow_removed = remove_shadow(image)
        clahe_enhanced = apply_clahe(shadow_removed)
        return clahe_enhanced
    
    else:
        clahe_enhanced = apply_clahe(image)
        enhanced = adaptive_threshold_enhance(clahe_enhanced)
        return enhanced

def color_enhance(image):
    """
    彩色图像增强
    :param image: 输入BGR图像
    :return: 增强后的BGR图像
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced
