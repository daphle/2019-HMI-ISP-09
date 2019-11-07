# 2019-HMI-ISP-09
e-book
import cv2
import numpy as np
from selenium import webdriver
import time
import argparse
from collections import deque
from pymouse import PyMouse
import pymouse
import pyautogui
from pathlib import Path


# 获取屏幕尺寸
screenWidth, screenHeight = pyautogui.size()
# 获取当前坐标位置
currentMouseX, currentMouseY = pyautogui.position()
# 将鼠标移动到屏幕中央
pyautogui.moveTo(screenWidth/2, screenHeight/2)

driver=webdriver.Chrome(r'C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe') 
#这是我的chromedriver的绝对路径，我是默认安装的，所以可以作为新手的参考
driver.get('http://www.baidu.com')

#通过ID找网页的标签，找到搜索框的标签    
seek_input=driver.find_element_by_id("kw")
#设置搜索的内容    
seek_input.send_keys("天气")
#找到搜索文档按钮    
seek_but=driver.find_element_by_id("su")
#并点击搜索 按钮    
seek_but.click()
#并点击搜索 按钮    
js="var q=document.documentElement.scrollTop=10000"
time.sleep(2)
driver.execute_script(js)
time.sleep(1)
total = 0  #页面数
#result = x.find_element_by_link_text("下一页>")
#result.click()
#time.sleep(2)

face_cascade = cv2.CascadeClassifier(
 'C:/Users/52426/Desktop/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
 'C:/Users/52426/Desktop/haarcascade_eye.xml')
eye_tree_eyeglasses_cascade = cv2.CascadeClassifier(
 'C:/Users/52426/Desktop/haarcascade_eye_tree_eyeglasses.xml')
profileface_cascade = cv2.CascadeClassifier(
 'C:/Users/52426/Desktop/haarcascade_profileface.xml')
# 笑脸检测器
smile_Cascade = cv2.CascadeClassifier('C:/Users/52426/Desktop/haarcascade_smile.xml')


page_right=False
page_left=False
eyes_open=True
eyes_close_count=0

second_x=screenWidth/2
second_y=screenHeight/2
second_w=0
first_x=screenWidth/2
first_y=screenHeight/2
first_w=0
count=0

capture = cv2.VideoCapture(0)
pyautogui.FAILSAFE=False
while(True):
    # 获取一帧
    ret, frame = capture.read()
    frame1 = cv2.flip(frame, 1)
    # 将这帧转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.flip(gray, 1)
    
    faces = face_cascade.detectMultiScale(gray1, 1.3, 5)
    profilefaces = profileface_cascade.detectMultiScale(gray, 2.1, 5)
    profileface_rights = profileface_cascade.detectMultiScale(gray1, 2.1, 5)
    
    l=len(faces)
    m = pymouse.PyMouse()   # 获取鼠标指针对象
    #print(m.position())    # 获取当前鼠标指针的坐标

    
    #检测左脸
    for (px,py,pw,ph) in profilefaces:
        page_right=True
        close_eyes=False
        cv2.rectangle(frame1,(640-px,py),(640-px-pw,py+ph),(255,193,193),2)
        cv2.putText(frame1, "right page", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        pyautogui.scroll(-50)
        #pyautogui.moveRel(10, None)

    #检测右脸
    for (rx,ry,rw,rh) in profileface_rights:
        page_left=True
        close_eyes=False
        cv2.rectangle(frame1,(rx,ry),(rx+rw,ry+rh),(153,50,204),2)
        cv2.putText(frame1, "left page", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        pyautogui.scroll(50)

    #利用训话读取每一张人脸的位置坐标
    for (x,y,w,h) in faces:
        count=count+1
        if count==1 :
            second_x=x+w/2
            second_y=y+h/2
            second_w=w
            count=0
        
        eyes_open=False
        #用矩形将人脸框出来 
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)  
        x2_x1 = second_x-first_x
        y2_y1 = second_y-first_y

        # 获取当前坐标位置
        currentMouseX, currentMouseY = pyautogui.position()
        if 0<(currentMouseX+3*x2_x1) < screenWidth and 0<(currentMouseY+3*y2_y1) < screenHeight :
            pyautogui.moveRel(3*x2_x1, 3*y2_y1)
        first_x=second_x
        first_y=second_y
        
        #页面缩放
        if (second_w-first_w) > 30 and first_w != 0:
            driver.fullscreen_window()
            cv2.putText(frame1, "max window", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        if (first_w-second_w) > 30 and second_w != 0:
            driver.set_window_size(500,500)
            cv2.putText(frame1, "min window", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)
        first_w=second_w
        
        roi_gray = gray1[y:y+h, x:x+w]
        roi_color = frame1[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        # 对人脸进行笑脸检测
        smile = smile_Cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.6,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # 框出上扬的嘴角并对笑脸打上Smile标签
        for (x2, y2, w2, h2) in smile:
            cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
            cv2.putText(frame1,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
            pyautogui.click()
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_gray1 = roi_gray[ey:ey+eh, ex:ex+ew]
            roi_color1 = roi_color[ey:ey+eh, ex:ex+ew]

            eye_tree_eyeglassess = eye_tree_eyeglasses_cascade.detectMultiScale(roi_gray1)

            for (cx,cy,cw,ch) in eye_tree_eyeglassess:
                cv2.circle(roi_color1,(cx+int(cw/2),cy+int(ch/2)),6,(255,255,0),2)
                eyes_open=True
                eyes_close_count=0
        
    if eyes_open is False :
        eyes_close_count=eyes_close_count+1
    #连续闭眼三帧
    if eyes_close_count == 3 :
        #截图
        i=1
        scrpath='C:/Users/52426/Desktop/screenshot'  #指定的保存目录
        capturename = '\\'+str(i) + '.png'  #自定义命名截图
        wholepath=scrpath+capturename
        if Path(scrpath).is_dir():  #判断文件夹路径是否已经存在
            pass    
        else: 
            Path(scrpath).mkdir()   #如果不存在，创建文件夹
        while Path(wholepath).exists():   #判断文件是否已经存在，也可使用is_file()判断
            i+=1
            capturename = '\\'+str(i) + '.png'
            wholepath = scrpath+capturename
        driver.get_screenshot_as_file(wholepath) #不能接受Path类的值，只能是字符串，否则无法截图
        #html = driver.page_source          #截图
        #driver.get_screenshot_as_file('C:/Users/52426/Desktop/screenshot.png')
        cv2.putText(frame1, "screenshot"+str(i), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    cv2.imshow('frame',frame1)
    #如果按键e则跳出本次循环
    k=cv2.waitKey(5)&0xFF
    if k == 27 :
        break
        
#摄像头释放  
camera.release()  
#销毁所有窗口  
cv2.destroyAllWindows()
