import cv2
import numpy as np

def preprocess_image(image, blur_ksize=(5, 5), canny_threshold1=50, canny_threshold2=150):
    """
    对输入图像进行预处理
    :param image: 输入图像 (BGR格式)
    :param blur_ksize: 高斯模糊核大小
    :param canny_threshold1: Canny边缘检测低阈值
    :param canny_threshold2: Canny边缘检测高阈值
    :return: 预处理后的二值边缘图
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    
    edges = cv2.Canny(blurred, canny_threshold1, canny_threshold2)
    
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return closed, gray

def enhance_edges(edges, iterations=2):
    """
    增强边缘，用于处理文档占比小或背景杂乱的情况
    :param edges: 边缘图像
    :param iterations: 形态学操作迭代次数
    :return: 增强后的边缘图像
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=iterations)
    eroded = cv2.erode(dilated, kernel, iterations=iterations)
    return eroded
