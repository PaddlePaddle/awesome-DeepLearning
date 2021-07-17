# coding:utf-8
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import qtawesome
import cv2
import random
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import qtawesome
from PyQt5.Qt import QMutex
from time import sleep, ctime
import numpy as np
import sys
import cv2
import imutils
sys.path.append('./')
from Detector import Detector
import random

#以下是OUT——GUI内容
class Initor_for_btn(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

    def init_bottom_box(self,res='检测结果'):
        self.bottom_box = QtWidgets.QLabel(self)
        self.bottom_box.setAlignment(Qt.AlignCenter)
        self.bottom_box.setStyleSheet("QLabel{background:black;}"
                                      "QLabel{color:rgb(255,255,255);"
                                      "font-size:25px;"
                                      "font-weight:bold;font-family:宋体;}"
                                      "border-radius: 25px;border: 1px solid black;")
        self.bottom_box.setText(res)
        self.bottom_box.setFixedSize(1700, 300)
        self.right_layout.addWidget(self.bottom_box, 9, 0, 1, 9)

    def init_right(self):

        # 原始视频
        self.raw_video = QtWidgets.QLabel(self)
        self.raw_video.setAlignment(Qt.AlignCenter)
        self.raw_video.setText("输入")
        self.raw_video.setFixedSize(800, 600)  # width height
        #self.raw_video.setFixedSize(self.video_size[0], self.video_size[1])  # width height
        self.raw_video.setStyleSheet("QLabel{background:black;}"
                                     "QLabel{color:rgb(255,255,255);"
                                     "font-size:25px;"
                                     "font-weight:bold;font-family:宋体;}"
                                     "border-radius: 25px;border: 1px solid black;")
        # 检测视频
        self.raw_video.move(290, 0)
        self.mask_video = QtWidgets.QLabel(self)
        self.mask_video.setAlignment(Qt.AlignCenter)
        self.mask_video.setText("输出")
        self.raw_video.setFixedSize(800, 600)  # width height
        #self.raw_video.setFixedSize(self.video_size[0], self.video_size[1])  # width height
        self.mask_video.setStyleSheet("QLabel{background:black;}"
                                      "QLabel{color:rgb(255,255,255);"
                                      "font-size:25px;"
                                      "font-weight:bold;font-family:宋体;}"
                                      "border-radius: 25px;border: 1px solid black;")
        self.right_bar_widget = QtWidgets.QWidget()  # 右侧顶部搜索框部件
        self.right_bar_layout = QtWidgets.QGridLayout()  # 右侧顶部搜索框网格布局
        self.right_bar_widget.setLayout(self.right_bar_layout)
        self.right_bar_layout.addWidget(self.raw_video, 0, 0)
        self.right_bar_layout.addWidget(self.mask_video, 0, 1)
        #addwidget6个参数表示控件名，行，列，占用行数，占用列数，对齐方式
        self.right_layout.addWidget(self.right_bar_widget, 0, 0, 1, 9)
        self.right_widget.setStyleSheet('''
            QWidget#right_widget{
                color:#232C51;
                background:white;
                border-top:1px solid darkGray;
                border-bottom:1px solid darkGray;
                border-right:1px solid darkGray;
                border-top-right-radius:10px;
                border-bottom-right-radius:10px;
            }
            QLabel#right_lable{
                border:none;
                font-size:25px;
                font-weight:700;
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            }
        ''')

    def init_left(self):

        self.left_close = QtWidgets.QPushButton(
            qtawesome.icon('fa.remove', color='white'), '')  # 关闭按钮
        self.left_mini = QtWidgets.QPushButton(
            qtawesome.icon('fa.minus', color='white'), '')  # 最小化按钮
        self.left_label_1 = QtWidgets.QPushButton("文件")
        self.left_label_1.setObjectName('left_label')
        self.left_button_1 = QtWidgets.QPushButton(
            qtawesome.icon('fa.film', color='white'), "图片")
        self.left_button_1.setObjectName('left_button')
        self.left_button_2 = QtWidgets.QPushButton(
            qtawesome.icon('fa.play', color='white'), "视频")
        self.left_button_2.setObjectName('left_button')
        self.left_button_6 = QtWidgets.QPushButton(
            qtawesome.icon('fa.comment', color='white'), "摄像头")
        self.left_button_6.setObjectName('left_button')
        self.left_button_1.setStyleSheet("font-size:25px")
        self.left_button_2.setStyleSheet("font-size:25px")
        self.left_button_6.setStyleSheet("font-size:25px")
        self.left_layout.addWidget(self.left_mini, 0, 0, 1, 1)
        self.left_layout.addWidget(self.left_close, 0, 2, 1, 1)
        self.left_layout.addWidget(self.left_label_1, 1, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_1, 2, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_2, 3, 0, 1, 3)
        self.left_layout.addWidget(self.left_button_6, 6, 0, 1, 3)
        self.left_widget.setStyleSheet('''
                QPushButton{border:none;color:white;}
                QPushButton#left_label{
                    border:none;
                    border-bottom:1px solid white;
                    color:white;
                    font-size:35px;
                    font-weight:1000;
                    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                }
                QPushButton#left_button:hover{border-left:4px solid red;font-weight:1500;}
            ''')

        self.left_close.setFixedSize(80, 80)  # 设置关闭按钮的大小
        #self.left_reset.setFixedSize(32, 32)  # 设置按钮大小
        self.left_mini.setFixedSize(80, 80)  # 设置最小化按钮大小

        self.left_close.setStyleSheet(
            '''QPushButton{background:#F76677;border-radius:7px;font-size:35px;}QPushButton:hover{background:red;}''')
      #  self.left_reset.setStyleSheet(
       #     '''QPushButton{background:#F7D674;border-radius:2px;}QPushButton:hover{background:yellow;}''')
        self.left_mini.setStyleSheet(
            '''QPushButton{background:#6DDF6D;border-radius:7px;font-size:35px;}QPushButton:hover{background:green;}''')

        self.left_widget.setStyleSheet(
            '''QWidget#left_widget{
                    background:black;
                    border-top:1px solid white;
                    border-bottom:1px solid white;
                    border-left:1px solid white;
                    border-top-left-radius:10px;
                    border-bottom-left-radius:10px;
                }'''
        )
#创建整个窗口的布局 不包括具体的控件 只是外部框框
    def init_ui(self):
        self.setFixedSize(1920, 1010)
        self.main_widget = QtWidgets.QWidget()  # 创建窗口主部件
        self.main_layout = QtWidgets.QGridLayout()  # 创建主部件的网格布局
        self.main_widget.setLayout(self.main_layout)  # 设置窗口主部件布局为网格布局
        self.left_widget = QtWidgets.QWidget()  # 创建左侧部件
        self.left_widget.setObjectName('left_widget')
        self.left_layout = QtWidgets.QGridLayout()  # 创建左侧部件的网格布局层
        self.left_widget.setLayout(self.left_layout)  # 设置左侧部件布局为网格
        self.right_widget = QtWidgets.QWidget()  # 创建右侧部件
        self.right_widget.setObjectName('right_widget')
        self.right_layout = QtWidgets.QGridLayout()
        self.right_widget.setLayout(self.right_layout)  # 设置右侧部件布局为网格
        self.main_layout.addWidget(
            self.left_widget, 0, 0, 12, 2)
        self.main_layout.addWidget(
            self.right_widget, 0, 2, 12, 10)#窗口右半边白色的部分
        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

class Initor_for_event(Initor_for_btn):

    def __init__(self):
        super().__init__()
        self.timer_camera = QTimer()  # 定义定时器

    def init_btn_event(self):
        self.left_mini.clicked.connect(self.showMinimized)
        self.left_button_1.clicked.connect(self.load_local_video_file)  # 点击选择文件

#以上是OUT-GUI内容

#以下是main_ui内容

class MainUi(Initor_for_event):
    '''
    工业缺陷检测功能界面
    '''

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.detector = Detector()
        self.setWindowTitle("基于深度学习的工业缺陷检测")
        self.timer = QTimer(self)  # 计时器，用于刷新界面
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        self.resize(1200, 910)
        self.camera_id = 0
        self.video_size = (500, 360)
        self.detectFlag = 0  # 初始不显示检测结果
        self.playFlag = 0  # 初始不显示摄像头画面
        # self.setFixedSize(self.width(), self.height())
        self.init_layout()
        self.main_layout.setSpacing(0)
        self.init_thread_params()
        self.playvideo = False
        self.video = None
        # 传参 lambda: self.btnstate(self.checkBox2)

    def init_thread_params(self):

        self.init_clik()

    def playon(self):
        self.playvideo = not self.playvideo
        videoName, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "*.avi;;*.mp4;;*.wmv;;All Files(*)")
        self.video = cv2.VideoCapture(videoName)

    def playshexiangtou(self):
        self.playvideo = not self.playvideo
        self.video = cv2.VideoCapture(0)

    def init_clik(self):

        self.left_close.clicked.connect(self.close_all)
        self.left_button_2.clicked.connect(self.playon)
        self.left_button_6.clicked.connect(self.playshexiangtou)

    def close_all(self):

        self.close()

    def init_play_btn(self):
        pass

    def init_layout(self):

        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
        self.init_left()
        self.init_right()
        self.init_bottom_box()
        self.init_btn_event()
        self.setWindowOpacity(0.99)  # 设置窗口透明度
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        self.main_layout.setSpacing(0)

    def load_local_video_file(self):
        
        videoName, _ = QFileDialog.getOpenFileName(self, "Open", "", "*.jpg;;*.png;;All Files(*)")
        res = "恭喜！未检测到缺陷！"
        if videoName != "":  # 为用户取消
            im_in = cv2.imdecode(np.fromfile(videoName, dtype=np.uint8), -1)
            im_in = imutils.resize(im_in, width=500)
            frame = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(frame.data, width, height, bytesPerLine,
                             QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
            self.raw_video.setPixmap(QPixmap.fromImage(q_image))
            im_out, boxes = self.detector.detect(im_in)
            detected_frame = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
            height, width, bytesPerComponent = detected_frame.shape
            bytesPerLine = bytesPerComponent * width
            q_image = QImage(detected_frame.data, width, height, bytesPerLine,
                             QImage.Format_RGB888).scaled(self.mask_video.width(), self.mask_video.height())
            self.mask_video.setPixmap(QPixmap.fromImage(q_image))
            if len(boxes)>=1:
                for xyxy, conf, cls in boxes:
                    x1, y1,x2, y2 = xyxy[:]
                    x_center, y_center = int((x1+x2)/2), int((y1+y2)/2)
                    res = "在({},{})处有{}缺陷，置信度为{:.2f}".format(x_center, y_center, cls, conf)
            self.init_bottom_box(res)

    def update(self):

        if self.playvideo and self.video:  # 为用户取消
            sucess, im_in = self.video.read()
            res = "恭喜！未检测到缺陷！"
            if sucess:
                
                im_in = imutils.resize(im_in, width=500)
                frame = cv2.cvtColor(im_in, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(frame.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.raw_video.width(), self.raw_video.height())
                self.raw_video.setPixmap(QPixmap.fromImage(q_image))

                im_out, boxes = self.detector.detect(im_in)
                detected_frame = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
                height, width, bytesPerComponent = detected_frame.shape
                bytesPerLine = bytesPerComponent * width
                q_image = QImage(detected_frame.data, width, height, bytesPerLine,
                                 QImage.Format_RGB888).scaled(self.mask_video.width(), self.mask_video.height())
                self.mask_video.setPixmap(QPixmap.fromImage(q_image))
                
                if len(boxes)>=1:
                    for xyxy, conf, cls in boxes:
                        x1, y1,x2, y2 = xyxy[:]
                        x_center, y_center = int((x1+x2)/2), int((y1+y2)/2)
                        res = "在({},{})处有{}缺陷，置信度为{:.2f}".format(x_center, y_center, cls, conf)
                self.init_bottom_box(res)
                
                
#以上是main_ui内容

#以下是GetfaceUI内容
class MonitorWindows(MainUi):
    '''
    实时检测界面
    '''

    def __init__(self):
        super().__init__()


if __name__ == "__main__":

    try:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        App = QApplication(sys.argv)
        monitor_box = MonitorWindows()
        monitor_box.show()
        sys.exit(App.exec_())
    except Exception as e:
        print(e)

    input('输入任意键退出')
#以上是GetfaceUI内容
MonitorWindows(MainUi)

