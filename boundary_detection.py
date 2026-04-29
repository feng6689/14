import cv2
import numpy as np

def find_largest_quadrilateral(edges, min_area_ratio=0.05):
    """
    在边缘图像中找到最大面积的近似四边形
    :param edges: 边缘图像
    :param min_area_ratio: 最小面积与图像总面积的比例阈值
    :return: 四边形的四个角点 (numpy array) 或 None，如果没有找到
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    image_area = edges.shape[0] * edges.shape[1]
    min_area = image_area * min_area_ratio
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    
    return None

def estimate_corners_with_hough(edges, gray_image):
    """
    使用霍夫线检测估算文档角点
    :param edges: 边缘图像
    :param gray_image: 灰度图像
    :return: 估算的四个角点 (numpy array)
    """
    h, w = edges.shape[:2]
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        return np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)
    
    horizontal_lines = []
    vertical_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        
        if abs(dx) > abs(dy):
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])
    
    has_horizontal = len(horizontal_lines) >= 2
    has_vertical = len(vertical_lines) >= 2
    
    if not has_horizontal or not has_vertical:
        return np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32)
    
    horizontal_lines = sorted(horizontal_lines, key=lambda l: (l[1] + l[3]) / 2)
    top_line = horizontal_lines[0]
    bottom_line = horizontal_lines[-1]
    
    vertical_lines = sorted(vertical_lines, key=lambda l: (l[0] + l[2]) / 2)
    left_line = vertical_lines[0]
    right_line = vertical_lines[-1]
    
    corners = []
    
    pt = line_intersection(top_line, left_line)
    if pt is not None:
        corners.append(pt)
    else:
        corners.append([0, 0])
    
    pt = line_intersection(top_line, right_line)
    if pt is not None:
        corners.append(pt)
    else:
        corners.append([w - 1, 0])
    
    pt = line_intersection(bottom_line, right_line)
    if pt is not None:
        corners.append(pt)
    else:
        corners.append([w - 1, h - 1])
    
    pt = line_intersection(bottom_line, left_line)
    if pt is not None:
        corners.append(pt)
    else:
        corners.append([0, h - 1])
    
    corners = np.array(corners, dtype=np.float32)
    corners = np.clip(corners, [0, 0], [w - 1, h - 1])
    
    return corners

def line_intersection(line1, line2):
    """
    计算两条线的交点
    :param line1: (x1, y1, x2, y2)
    :param line2: (x1, y1, x2, y2)
    :return: 交点 (x, y) 或 None
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    if 0 <= t <= 1 and 0 <= u <= 1:
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        return [x, y]
    return None

def get_default_corners(image_shape):
    """
    获取默认角点（图像的四个顶点）
    :param image_shape: 图像形状 (h, w)
    :return: 四个默认角点
    """
    h, w = image_shape[:2]
    return np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
