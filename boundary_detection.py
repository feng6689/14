import cv2
import numpy as np

def is_valid_quadrilateral(approx, image_shape):
    """
    检查四边形是否有效
    :param approx: 近似轮廓 (4个点)
    :param image_shape: 图像形状 (h, w)
    :return: (是否有效, 面积)
    """
    if len(approx) != 4:
        return False, 0
    
    h, w = image_shape[:2]
    image_area = h * w
    
    area = cv2.contourArea(approx)
    
    if area < image_area * 0.02:
        return False, area
    
    if area > image_area * 0.95:
        return False, area
    
    pts = approx.reshape(4, 2).astype(np.float32)
    
    (tl, tr, br, bl) = order_points(pts)
    
    width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.sqrt(((bl[0] - tl[0]) ** 2) + ((bl[1] - tl[1]) ** 2))
    height_right = np.sqrt(((br[0] - tr[0]) ** 2) + ((br[1] - tr[1]) ** 2))
    max_height = max(int(height_left), int(height_right))
    
    aspect_ratio = max(max_width, max_height) / max(min(max_width, max_height), 1)
    
    if aspect_ratio > 10:
        return False, area
    
    return True, area

def order_points(pts):
    """
    对点进行排序：左上、右上、右下、左下
    """
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def find_document_boundary(edges, image_shape):
    """
    在边缘图像中查找文档边界
    :param edges: 边缘图像
    :param image_shape: 图像形状
    :return: (四个角点, 面积) 或 (None, 0)
    """
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, 0
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    best_quad = None
    best_area = 0
    
    for contour in contours[:10]:
        peri = cv2.arcLength(contour, True)
        
        for epsilon in [0.01, 0.02, 0.03, 0.04, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            
            if len(approx) == 4:
                valid, area = is_valid_quadrilateral(approx, image_shape)
                
                if valid and area > best_area:
                    best_quad = approx
                    best_area = area
    
    if best_quad is not None:
        points = best_quad.reshape(4, 2).astype(np.float32)
        return points, best_area
    
    return None, 0

def find_document_with_multiple_strategies(preprocess_results, image_shape):
    """
    使用多种预处理策略查找文档边界
    :param preprocess_results: 多种预处理结果
    :param image_shape: 图像形状
    :return: (四个角点, 是否使用估算) 或 (None, False)
    """
    best_corners = None
    best_area = 0
    
    for strategy_name, edges, gray in preprocess_results:
        corners, area = find_document_boundary(edges, image_shape)
        
        if corners is not None and area > best_area:
            best_corners = corners
            best_area = area
    
    if best_corners is not None:
        return best_corners, False
    
    print("尝试增强边缘后再次检测...")
    for strategy_name, edges, gray in preprocess_results:
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        eroded = cv2.erode(dilated, kernel, iterations=1)
        
        corners, area = find_document_boundary(eroded, image_shape)
        
        if corners is not None and area > best_area:
            best_corners = corners
            best_area = area
    
    if best_corners is not None:
        return best_corners, False
    
    print("未找到清晰边界，尝试霍夫线检测...")
    corners = estimate_corners_with_hough_lines(preprocess_results[0][2], image_shape)
    
    if corners is not None:
        print("未找到清晰边界，采用估算角点")
        return corners, True
    
    return None, False

def estimate_corners_with_hough_lines(gray, image_shape):
    """
    使用霍夫线检测估算角点
    """
    h, w = image_shape[:2]
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    lines = cv2.HoughLinesP(dilated, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=20)
    
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
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if angle < 30:
                horizontal_lines.append(line[0])
        else:
            angle = abs(np.arctan2(dy, dx) * 180 / np.pi)
            if angle > 60:
                vertical_lines.append(line[0])
    
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
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
    corners.append(pt if pt is not None else [0, 0])
    
    pt = line_intersection(top_line, right_line)
    corners.append(pt if pt is not None else [w - 1, 0])
    
    pt = line_intersection(bottom_line, right_line)
    corners.append(pt if pt is not None else [w - 1, h - 1])
    
    pt = line_intersection(bottom_line, left_line)
    corners.append(pt if pt is not None else [0, h - 1])
    
    corners = np.array(corners, dtype=np.float32)
    corners = np.clip(corners, [0, 0], [w - 1, h - 1])
    
    return corners

def line_intersection(line1, line2):
    """
    计算两条线的交点（支持线段外延长线相交）
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    x = int(x1 + t * (x2 - x1))
    y = int(y1 + t * (y2 - y1))
    
    return [x, y]
