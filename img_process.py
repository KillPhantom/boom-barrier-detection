import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time

# start = time.time()# 计算程序运算时间

# 大致思路就是 
#1.在杆静止的时候截取图像下部的横杆所在区域
#2.区域进行统计霍夫检测，返回一系列线段，取杆平放时候的x，y坐标的最小值
#3.不断调试参数，利用每帧图像的x,y坐标的最小值，进行判断杆是否抬起
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

            




for i in range (2,326):
# if __name__ =='__main__':

	image=cv2.imread('people/%d.jpg' %i)
	bot_left = [630, 1065]
	bot_right = [1360,1500]
	apex_right = [1360, 600]
	apex_left = [630, 600]
	#上图即是截取区域
	v = [np.array([bot_left, bot_right, apex_right, apex_left], dtype=np.int32)]
	gray = grayscale(image)#灰度变换
	blur = gaussian_blur(gray, 29)#高斯模糊
	edge = canny(blur, 50, 120)# 边缘检测
	mask = region_of_interest(edge, v)#取roi
	lines = cv2.HoughLinesP(mask, 0.8, np.pi/180, 25,minLineLength=60,maxLineGap=6)#统计霍夫变换
	font=cv2.FONT_HERSHEY_SIMPLEX #打印到图片上的默认字体

	#下面代码是将所有线段两点坐标进行排序比较，选出x和y 的最小值
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
	print(i)
	# 下面是判断条件
	if lines is None or min_y<598 or min_y>980: #你打开张图看看就知道了 
		
		print('杆已抬起')
		#img=cv2.putText(image,'lifted',(40,100),font,2.5,(0,0,255),5) #写字，参数百度
		#cv2.imwrite('people_result/result%d.jpg' %i,img)

	elif abs(933-min_x)>200:
		print('杆已抬起')
		#img=cv2.putText(image,'lifted',(40,100),font,2.5,(0,0,255),5)
		#cv2.imwrite('people_result/result%d.jpg' %i,img)		
		# draw_lines(image, lines, thickness=10)
		# plt.imshow(image)
		# plt.show()
		# font=cv2.FONT_HERSHEY_SIMPLEX
		# img=cv2.putText(image,'1',(40,100),font,2.5,(0,0,255),5)
		# cv2.imwrite('result.jpg',img)

	else:
		print('杆未抬起')
		#img=cv2.putText(image,'falls down',(40,100),font,2.5,(0,0,255),5)
		#cv2.imwrite('people_result/result%d.jpg' %i,img)		
		#line_img = np.copy((image)*0)
		#print(lines)



		# draw_lines(image, lines, thickness=10)
		# plt.imshow(image)
		# plt.show()

# 进一步修正


	# 计时代码 ：
	# end = time.time()
	# cost =( end-start)
	# print('finished and take %f second'%cost )