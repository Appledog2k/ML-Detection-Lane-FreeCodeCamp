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

# đọc ảnh dưới dạng mảng đa chiều chứa cường độ ảnh tương đối của từng pixels
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)


plt.imshow(canny)
plt.show()