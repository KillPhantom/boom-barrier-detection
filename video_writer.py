import cv2

fps = 24   #视频帧率
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') #编码规则

videoWriter = cv2.VideoWriter('Result6.avi', fourcc, fps, (1920,1080))   #(1920，1080)为视频大小
for i in range(2,376):
    img12 = cv2.imread('night2_result/result%d.jpg' %i)

#    cv2.waitKey(1000/int(fps)) #这个延长视频时间
    videoWriter.write(img12)
    print('%d has done' %i)# 标识当前帧写入完毕
videoWriter.release()
print('finished')