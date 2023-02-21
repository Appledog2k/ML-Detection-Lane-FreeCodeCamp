import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))         # slightly lower than the middle
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image,lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    left_fit_average = np.average(left_fit,axis=0)
    right_fit_average = np.average(right_fit,axis=0)
    left_line =  make_coordinates(image, left_fit_average)
    right_line =  make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# 2 đối số thứ nhất ảnh đầu vào đối số thứ 2 sử dụng cờ xám để chuyển đôiỉ hình ảnh sang thang độ xám đầu ra gray là một ảnh
# dùng gaussian làm mờ làm mịn ảnh với hạt nhân là một ma trận 5x5  và độ lệch bằng 0
# dùng thuật toán canny(xác định hình ảnh) giúp xác định cạnh


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    canny = cv2.Canny(blur, 50,150)
    return canny

# định  nghĩa một vùng quan tâm, vì vùng bao quanh có hình dạng tam giác 
# khai báo một mảng có nhiều mảng, bên trong mảng chỉ định các đỉnh của nó
# khai báo một mặt có cùng số pixel với ảnh ban dầu

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            # print(line)
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
    [(200,height),(1100,height),(550,250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    # Xử dụng tính chất việc nhân nhị phân, hàm tích nhị phân
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image



# Thuật toán xác định cạnh, xác định ranh giới của các đối tượng trong hình ảnh, tìm các vùng ảnh có sự thay đổi rõ rệt về cường độ,
# màu sắc, một hình ảnh có thể được đọc dưới dạng một ma trận mảng pixels một pixels chứa cường độ ánh sáng taị một số vị trí trong ảnh
# môi cường độ pixels được biểu thj bằng một giá trị số nẳm trong khoảng từ 0 đến 255

# Các bước tiến hành:(Convert image to grayscale) Chuyển đổi hình ảnh sang thang độ xám ( hình ảnh được tạo thành tử các pixel, hình ảnh màu 3 kênh sẽ có màu xanh lục
# đỏ và các kênh màu xanh lam, mỗi pixcel là sự kết hợp của 3 giá trị cường độ, trong khi hình ảnh thang độ xám chỉ có một kênh, mỗi pixel chỉ có một giá tri cường độ nằm trong khoangr
# 0 tới 255, bằng cách sử dụng hình ảnh thang độ xám sử lý một kênh nhanh hơn xử lý 3 kênh hình ảnh màu và giúp giảm bớt tính toán chuyên sâu
# (reduce noise) giảm nhiễu và làm mịn ảnh (do nhiễu có thể tạo ra các cạnh giả ảnh hưởng tới khả năng phát hiện cạnh)
# để giảm nhiễu làm mịn chúng ta sử dụng bộ lọc gaussian : mootk bộ lọc được lưu trữ dưới dạng một tập hợp pixel kín, mỗi pixels
# cho một hình ảnh thang độ xám được biểu thị bằng một số nằm trong giải giá trị. Giả sử có một pixcel cần làm mờ,mịn phương pháp là
# sửa đổi giá trị pixel bằng trung binhf của cường độ pixel xung quanh nó (thông thuiowngf được sử dụng hạt nhân giúp đặt giá trị pixel bằng giá trị trung bình
# pixels lân cận)
# Phát hiện hình ảnh bằn thuật toán (Canny edge detector) 1986: có thể biểu diễn hình ảnh dưới dạng không gian tọa độ 2 chiều, trục x đi qua chiều rộng, trục y đi qua chiều cao
# sao cho tích x và y mang lại tổng số pixels, ngoài ra không chỉ xem hình ảnh dưới dạng một mảng mà có thể xem dưới dạng một hàm liên tục
# x va y đạo hàm theo 2 hướng x và y từ đó bằng cách so sánh sự thanh đổi về cường độ từ các pixel liền kề, một đạo hàm nhỏ một thay đổi nhỏ
# về cường độ, một đạo hàm lớn một thay đổi lướn về cường độ theo mọi hướng của hình ảnh, khi gọi hàm canny sẽ tính toán độ dốc về mọi hướng
# theo hình ảnh đầu vào sau đó theo dõi độ dốc mạnh nhất dưới dạng một loạt pixels trắng, nếu độ dốc lớn hơn ở ngưỡng treewn thì được chấp nhận
# là pixel cạnh và nếu ở ngướng dưới thì bị từ chối nếu nằm giữa 2 ngưỡng thì chỉ được chấp nhận nêus đước kết nối  với một canh "mạnh"
# có tỉ lệ 1-2 or 1-3
# thuật toán hough giúp xác định hình dạng, xác định hình dạng thông qua mathplot cô lập hình ảnh


# đọc ảnh trả về dưới dạng một mảng đa chiều chứa cường độ của từng fixel
image = cv2.imread('test_image.jpg')

# tính tham chiếu, tham trị nên phải tạo bản sao có phương thức copy
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)

# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,100, np.array([]), minLineLength=40, maxLineGap=5)
# averages_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image,averages_lines)
# combo_image= cv2.addWeighted(lane_image, 0.8,line_image,1,1)
# cv2.imshow('result', combo_image)

# cv2.waitKey(0)
# plt.imshow(canny)
# plt.show()

cap = cv2.VideoCapture("test_video.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180,100, np.array([]), minLineLength=40, maxLineGap=5)
    averages_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame,averages_lines)
    combo_image= cv2.addWeighted(frame, 0.8,line_image,1,1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()