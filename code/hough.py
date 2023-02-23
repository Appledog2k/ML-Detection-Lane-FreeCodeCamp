import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    # Sử dụng gaussian blur để giảm độ nhiễu loại bỏ cạnh giả
    blur = cv2.GaussianBlur(lane_image,(5,5),0)
    # chuyển đổi thành ảnh thang độ xám với một kênh màu duy nhất
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)
    # thuật toán phát hiện cạnh Canny
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # print(line)
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image
# lấy về phân vùng cần xác định hình dạng
def region_of_interest(image):
    polygons = np.array([
    [(300, 530),(1050, 530), (630,260)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # lấy lại hình trong phân vùng bằng tính chất nhị phân
    return masked_image

# đọc ảnh dưới dạng mảng đa chiều chứa cường độ ảnh tương đối của từng pixels
image = cv2.imread('anh1.png')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
# hough-line
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=70)
line_image = display_lines(lane_image, lines)
cv2.imshow('result', line_image)
cv2.waitKey(0)