import cv2

vc=cv2.VideoCapture("night_short2.wmv") #读取视频
c=1 #帧数
if vc.isOpened():  
    rval,frame=vc.read()  #返回值自己查，frame 是当前帧
else:  
    rval=False  
while rval:  
    rval,frame=vc.read()
    c+=1 #帧数不断增加
    rows, cols, channel = frame.shape #返回图像的三维
    # frame2=cv2.resize(frame,(cols/3,rows/3),fx=0,fy=0,interpolation=cv2.INTER_AREA) # 重新resize图像大小
    #if ( c%5 == 0):                            #每隔5帧取一帧检测
    cv2.imwrite('night2/%d.jpg' %c,frame) #将图像写入到指定目录下

        # cropped001 = frame2[0:300,300:600]   #裁剪图像，你用不上
        # cv2.imwrite('./cropped/'+str(c)+'_001.jpg',cropped001)
    print(c) 
    cv2.waitKey(1)  
vc.release() 