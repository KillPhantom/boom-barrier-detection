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

            


for i in range (1,224):
# if __name__ =='__main__':
	image=cv2.imread('day1_frame/%d.jpg' %i)
	# image=cv2.imread('day1_frame/150.jpg')

	bot_left = [630, 1065]
	bot_right = [1360,1500]
	apex_right = [1360, 520]
	apex_left = [630, 520]
	v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
	gray = grayscale(image)
	blur = gaussian_blur(gray, 25)
	edge = canny(blur, 50, 130)

	mask = region_of_interest(edge, v)
	lines = cv2.HoughLinesP(mask, 0.8, np.pi/180, 25,minLineLength=60,maxLineGap=6)


	if not lines is None:
		temp=[]
		temp2=[]
		for line in lines:
			for x1,y1,x2,y2 in line:
				temp.append(abs(y2-y1)/abs(x2-x1))

		ave = np.mean(temp)
	print(ave)
		# draw_lines(image, lines, thickness=10)
		# plt.imshow(image)
		# plt.show()

	if not lines is None and 1<ave<2:
		print(i,'杆落下')
	else:
		print(i,'杆抬起') 



