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

# lấy về phân vùng cần xác định hình dạng
def region_of_interest(image):
    polygons = np.array([
    [(0, 500),(1050, 700), (600,200)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask) # lấy lại hình trong phân vùng bằng tính chất nhị phân
    return masked_image

# đọc ảnh dưới dạng mảng đa chiều chứa cường độ ảnh tương đối của từng pixels
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

cropped_image = region_of_interest(canny)
cv2.imshow('result', region_of_interest(cropped_image))
cv2.waitKey(0)