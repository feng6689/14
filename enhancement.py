import cv2
import numpy as np

def remove_shadow_gray(gray):
    """
    移除灰度图像中的阴影，保留细节
    :param gray: 灰度图像
    :return: 去除阴影后的灰度图像
    """
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    
    background = cv2.medianBlur(dilated, 21)
    
    diff = cv2.subtract(background, gray)
    
    result = 255 - diff
    
    return result

def apply_clahe_gray(gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    对灰度图像应用CLAHE
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)

def enhance_gray_document(image):
    """
    增强灰度文档（推荐用于扫描件）
    不会将阴影变成黑色
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    shadow_removed = remove_shadow_gray(gray)
    
    clahe_enhanced = apply_clahe_gray(shadow_removed)
    
    return clahe_enhanced

def enhance_color_document(image):
    """
    增强彩色文档
    保留彩色信息，改善光照不均
    """
    if len(image.shape) != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    l_shadow_removed = remove_shadow_gray(l)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_shadow_removed)
    
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    color_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return color_enhanced

def create_scan_like_binarization(image):
    """
    创建类扫描的二值化效果（清晰文字）
    这个方法不会把阴影变成黑色，而是先去除阴影再二值化
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    shadow_removed = remove_shadow_gray(gray)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(shadow_removed)
    
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15, 8
    )
    
    return binary

def white_balance(image):
    """
    简单的白平衡校正
    """
    if len(image.shape) != 3:
        return image
    
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    return result

def enhance_document(image, method='balanced'):
    """
    综合文档增强方法
    :param image: 输入图像 (BGR)
    :param method: 增强方法
        - 'gray': 灰度增强（CLAHE + 阴影去除）
        - 'color': 彩色增强（保留色彩）
        - 'binary': 二值化（类扫描效果，文字清晰）
        - 'balanced': 平衡方案（推荐，灰度但保留细节）
    :return: 增强后的图像
    """
    if method == 'gray':
        return enhance_gray_document(image)
    
    elif method == 'color':
        color_result = enhance_color_document(image)
        return white_balance(color_result)
    
    elif method == 'binary':
        return create_scan_like_binarization(image)
    
    elif method == 'balanced':
        return enhance_gray_document(image)
    
    else:
        return enhance_gray_document(image)

def get_enhancement_methods():
    """
    返回可用的增强方法列表
    """
    return {
        'gray': '灰度增强 - 去阴影 + CLAHE',
        'color': '彩色增强 - 保留色彩，改善光照',
        'binary': '二值化 - 类扫描效果，文字清晰',
        'balanced': '平衡方案 - 推荐，灰度但保留细节'
    }
