import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
start = time.time()
#   


import math

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):

    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            

# def hough_lines(img, rho, theta, threshold):

#     lines = cv2.HoughLines(img, rho, theta, threshold)
#     line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
#     return line_img



if __name__ =='__main__':

	image=cv2.imread('people/167.jpg')
	print(image.shape)
	bot_left = [840, 410]
	bot_right = [940,410]
	apex_right = [940, 0]
	apex_left = [840, 0]
	v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
	gray = grayscale(image)
	blur = gaussian_blur(gray, 15)
	edge = canny(blur, 50, 120)

	mask = region_of_interest(edge, v)
	lines = cv2.HoughLinesP(mask, 0.8, np.pi/180, 25,minLineLength=60,maxLineGap=6)


	if not lines is None:
		temp=[]
		temp2=[]
		for line in lines:
			for x1,y1,x2,y2 in line:
				temp.append(x1)
				temp.append(x2)
				temp2.append(y1)
				temp2.append(y2)
		min_x = min(temp)
		min_y = min(temp2)
		max_y = max(temp2)

		print(min_y)
	if lines is None or min_y>120 :
		
		print('杆未抬起')
		draw_lines(image, lines, thickness=10)
		plt.imshow(image)
		plt.show()

	# elif abs(933-min_x)>200:
	# 	print('杆已抬起')
	# 	mark.append(1)
	# 	draw_lines(image, lines, thickness=10)
	# 	plt.imshow(image)
	# 	plt.show()
		# font=cv2.FONT_HERSHEY_SIMPLEX
		# img=cv2.putText(image,'1',(40,100),font,2.5,(0,0,255),5)
		# cv2.imwrite('result.jpg',img)

	else:
		print('杆已抬起')

		#line_img = np.copy((image)*0)
		#print(lines)



		draw_lines(image, lines, thickness=10)
		plt.imshow(image)
		plt.show()

# 进一步修正


	# 计时代码 ：
	# end = time.time()
	# cost =( end-start)
	# print('finished and take %f second'%cost )