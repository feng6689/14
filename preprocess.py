import cv2
import numpy as np

def preprocess_image_multi_strategy(image):
    """
    多种预处理策略，提高边界检测成功率
    :param image: 输入BGR图像
    :return: 多种预处理结果的列表
    """
    results = []
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred1 = cv2.GaussianBlur(gray, (5, 5), 0)
    edges1 = cv2.Canny(blurred1, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    closed1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel, iterations=2)
    results.append(('canny_standard', closed1, gray))
    
    blurred2 = cv2.GaussianBlur(gray, (7, 7), 0)
    edges2 = cv2.Canny(blurred2, 30, 100)
    kernel2 = np.ones((5, 5), np.uint8)
    closed2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel2, iterations=3)
    results.append(('canny_soft', closed2, gray))
    
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    kernel3 = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(thresh, kernel3, iterations=2)
    results.append(('adaptive_thresh', dilated, gray))
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    v_edges = cv2.Canny(v_blur, 30, 100)
    results.append(('hsv_value', v_edges, gray))
    
    return results

def enhance_edges_strong(edges):
    """
    强边缘增强
    """
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    return eroded
