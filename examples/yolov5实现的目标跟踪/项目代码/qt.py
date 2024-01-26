import sys
sys.path.insert(0, './yolov5')

from PyQt5.QtWidgets import *                   #这两个是pyqt5常用的库
from PyQt5.QtGui import QIcon, QPixmap          #可以满足小白大多数功能

import os
import track


# 创立一个主界面，并保持它，从各种按钮或者组件中接受信号完成界面功能，相当于无限循环
# 只有选择退出后才会关掉程序退出循环

class MyClass(QWidget):
    def __init__(self):
        super().__init__()     #继承父类
        self.initUI()          #自己定义的函数，初始化类界面，里面放着自己各种定义的按钮组件及布局
        self.child_window = ChildClass()       #子界面的调用，本质和主界面一样是一个类，在这里将其声明为主界面的成员


def initUI(self):
    self.setWindowTitle("COC缺陷检测")  # 设置界面名称


    # self.setWindowIcon(QIcon("iconimg/zhou.png"))        #设计界面的图标，图片放在项目文件夹的子文件夹里就不会出错，名字也要对应
    self.resize(350, 200)  # 设置界面大小
    self.TModelSelectSignal = [0, 0]  # 选择按钮对应的模型
    self.TModel = [0, 0]  # 表示已经训练好的模型编号

    myframe = QFrame(self)  # 实例化一个QFrame可以定义一下风格样式，相当于一个框架，可以移动，其内部组件也可以移动
    btn2 = QPushButton("开始训练模型", self)  # 定义一个按钮，括号里需要一个self，如果需要在类内传递，则应该定义为self.btn2
    btn2.clicked.connect(self.TestModel)  # 将点击事件与一个函数相连，clicked表示按钮的点击事件，还有其他的功能函数，后面连接的是一个类内函数，调用时无需加括号
    btn3 = QPushButton("上传数据集", self)
    btn3.clicked.connect(self.DataExplorerSelect)  # 连接一个选择文件夹的函数
    btn5 = QPushButton("退出程序", self)
    btn5.clicked.connect(self.close)  # 将按钮与关闭事件相连，这个关闭事件是重写的，它自带一个关闭函数，这里重写为点击关闭之后会弹窗提示是否需要关闭
    btn6 = QPushButton("检测", self)
    btn6.clicked.connect(self.show_child)  # 这里将联系弹出子界面函数，具体弹出方式在函数里说明

    combol1 = QComboBox(myframe)  # 定义为一个下拉框，括号里为这个下拉框从属的骨架（框架)
    combol1.addItem("   选择模型")  # 添加下拉选项的文本表示，这里因为没有找到文字对齐方式，所以采用直接打空格，网上说文字对齐需要重写展示函数
    combol1.addItem("   YOLOv3")
    combol1.addItem("   YOLOv4")
    combol1.activated[str].connect(self.TModelSelect)  # |--将选择好的模型序号存到模型选择数组里
    # |--后面的训练函数会根据这个数组判断需要训练哪个模型
    # |--[str]表示会将下拉框里的文字随着选择信号传过去
    # |--activated表示该选项可以被选中并传递信号
    vlo = QVBoxLayout()  # 创建一个垂直布局，需要将需要垂直布局的组件添加进去
    vlo.addWidget(combol1)  # 添加相关组件到垂直布局里
    vlo.addWidget(btn3)
    vlo.addWidget(btn2)
    vlo.addWidget(btn6)
    vlo.addWidget(btn5)
    vlo.addStretch(1)  # 一个伸缩函数，可以一定程度上防止界面放大之后排版不协调
    hlo = QVBoxLayout(self)  # 创建整体框架布局，即主界面的布局
    hlo.addLayout(vlo)  # 将按钮布局添加到主界面的布局之中
    hlo.addWidget(myframe)  # 将框架也加入到总体布局中，当然也可以不需要这框架，直接按照整体框架布局来排版
    # 之所以这里有这个myframe，是因为尝试过很多种布局，其中一个布局就是将其他组件都放到这个myframe中，移动这个myframe
    # 其里面的组件布局相对位置不会改变，后面又尝试了多种布局，所以这个myframe最后里面其实就剩下一个下拉框
    self.show()  # 显示主界面

def DataExplorerSelect(self):
    path = r'D:\pycharm\QTYOLOV3\yolov3\VOCdevkit\VOC2007'
    os.system("explorer.exe %s" % path)


def show_child(self):
    TModel1 = self.TModel  # |--这是子界面的类内函数


    self.child_window.GetTModel(TModel1)  # |--将训练好的模型序号传到子界面的类内参数里面
    self.child_window.show()  # |--子界面相当于主界面的一个类内成员
    # |--但是本质还是一个界面类，也有show函数将其展示

def TModelSelect(self, s):  # s是形参，表示传回来的选中的选项的文字
    if s == '   YOLOv3':
        self.TModelSelectSignal[0] = 1  # 如果选中的是YOLOv3-COC就将第一位置1
        # print(self.TModelSelectSignal[0])
    elif s == '   YOLOv4':
        self.TModelSelectSignal[1] = 1  # 如果选中的是YOLO-Efficientnet就将第二位置1
        # print(self.TModelSelectSignal[1])


def TestModel(self):
    if self.TModelSelectSignal[0] == 1:
        train.run()
        self.TModelSelectSignal[0] = 0
    else:
        print("没有该模型")

def closeEvent(self, event):
    result = QMessageBox.question(self, "提示：", "您真的要退出程序吗", QMessageBox.Yes|QMessageBox.No, QMessageBox.Yes)
    if result == QMessageBox.Yes:
        event.accept()
    else:
        event.ignore()

class ChildClass(QWidget):
      def __init__(self):
          super().__init__()
          self.initUI()
          self.TModel = []  # 用来接收主界面的训练好的模型的序号
          self.openfile_name_image = ''  # 存储原始图像的地址
          self.result_name_image = ''  # 存储检测好的图像的地址

def initUI(self):
    self.resize(1100, 450)	#缩放界面大小
    self.setWindowTitle("目标检测")	#设置界面标题
 # self.setWindowIcon(QIcon("iconimg/zhou.png"))		#设置界面图标
    self.PModelSelectSignal = [0, 0]	#设置需要预测模型的序号，在下拉框里选择

    myframe = QFrame(self)
    self.label1 = QLabel("检测模型", self)
    combol1 = QComboBox(myframe)
    combol1.addItem("选择检测模型")
    combol1.addItem("YOLOV3")
    combol1.addItem("YOLOV4")
    combol1.activated[str].connect(self.PModelSelect)	#链接预测模型序号选择函数
    btn1 = QPushButton("选择检测图片", self)
    btn1.clicked.connect(self.select_image)		#链接检测图片选择函数，本质是打开一个文件夹

    btn2 = QPushButton("开始检测", self)
    btn2.clicked.connect(self.PredictModel)		#链接预测模型函数

    self.label2 = QLabel("", self) 			#创建一个label，可以存放文字或者图片，在这里是用来存放图片，文本参数为空就会显示为空，留出空白区域，选择好图片时会有函数展示图片
    self.label2.resize(400, 400)
    self.label3 = QLabel("", self)
    self.label3.resize(400, 400)
    label4 = QLabel("      原始图片", self)		#用来放在图片底部表示这是哪一种图片
    label5 = QLabel("      检测图片", self)
    vlo2 = QHBoxLayout()		#创建一个子布局，将图片水平排放
    vlo2.addWidget(label4)
    vlo2.addWidget(label5)

    vlo = QHBoxLayout()		#创建一个子布局，将按钮水平排放
    vlo.addStretch()
    vlo.addWidget(self.label1)
    vlo.addWidget(combol1)
    vlo.addWidget(btn1)
    vlo.addWidget(btn2)
    vlo.addStretch(1)

    vlo1 = QHBoxLayout()	#创建一个水平布局，将两个提示标签竖直排放
    vlo1.addWidget(self.label2)
    vlo1.addWidget(self.label3)

    hlo = QVBoxLayout(self)		#创建一个总的垂直布局，将三个子布局垂直排放
    hlo.addLayout(vlo)
    hlo.addLayout(vlo1)
    hlo.addStretch(1)
    hlo.addLayout(vlo2)
    hlo.addStretch(0)
    hlo.addWidget(myframe)

def GetTModel(self, a):
    self.TModel = a

def closeEvent(self, event):
    result = QMessageBox.question(self, "提示：", "您真的要退出程序吗", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if result == QMessageBox.Yes:
        event.accept()
    else:
        event.ignore()

def select_image(self):
    self.openfile_name_image, _ = QFileDialog.getOpenFileName(self, "选择照片文件",
        r"./yolov3/imgtest/")
 	#弹出一个对话窗，是一个文件夹，可以选择一个文件然后返回地址到 self.openfile_name_image中
    print('加载照片文件地址为：' + str(self.openfile_name_image))
    self.label2.setPixmap(QPixmap(str(self.openfile_name_image))) #将选中的文件名字传入QPixmap（）中，括号内为文件地址，就会读取这个图片
    self.label2.resize(300, 400)
    self.label2.setScaledContents(True)	#表示这个label可以可以自适应窗口大小，可以让图片随窗口大小而变化

def PModelSelect(self, s):
    if s == 'YOLOV3':
        if self.TModel[0] == 1:
            self.PModelSelectSignal[0] = 1
            self.PModelSelectSignal[1] = 0
            print(self.PModelSelectSignal[0])
        else:
            print("模型YOLOV3未训练")	##如果已经训练好的模型数组里对应的位置为0，则表示该模型未训练
            self.PModelSelectSignal[1] = 0		#同时也要讲模型选择信号清零，以便下次可以继续选择赋值
    elif s == 'YOLOV4':
        if self.TModel[1] == 1:
            self.PModelSelectSignal[1] = 1
            self.PModelSelectSignal[0] = 0
            print(self.PModelSelectSignal[1])
        else:
            print("模型YOLOV4未训练")
            self.PModelSelectSignal[0] = 0

def PredictModel(self):
    if self.PModelSelectSignal[0] == 1:
        def PredictModel(self):

            if self.PModelSelectSignal[0] == 1:
                predict.predict(self.openfile_name_image)  # 将需要预测的图片传入导入的预测函数
            elif self.PModelSelectSignal[1] == 1:
                print('YOLOV4正在检测')  # 这里应该放入另外一个模型
            else:
                print('没有该模型')
        a = self.openfile_name_image
        a = a.split('/')  # 将预测图片里的编号分离出来
        a = './yolov3/imgtestresult/' + a[-1]  # 将指定路径与图片编号组合，即可得到预测好的图片的路径
        self.label3.setPixmap(QPixmap(a))  # 直接读取预测好的图片
        self.label3.resize(300, 400)
        self.label3.setScaledContents(True)
        print(a)
        predict.predict(self.openfile_name_image)	#将需要预测的图片传入导入的预测函数
    elif self.PModelSelectSignal[1] == 1:
        print('YOLOV4正在检测') #这里应该放入另外一个模型
    else:
        print('没有该模型')
    a = self.openfile_name_image
    a = a.split('/')	#将预测图片里的编号分离出来
    a = './yolov3/imgtestresult/' + a[-1]	#将指定路径与图片编号组合，即可得到预测好的图片的路径
    self.label3.setPixmap(QPixmap(a))	#直接读取预测好的图片
    self.label3.resize(300, 400)
    self.label3.setScaledContents(True)
    print(a)

# 创立一个主界面，并保持它，从各种按钮或者组件中接受信号完成界面功能，相当于无限循环
# 只有选择退出后才会关掉程序退出循环
if __name__ == '__main__':
    app = QApplication(sys.argv)
    mc = MyClass()		#这里相当于实例化一个主界面，myclass是自己定义的主界面类
    sys.exit(app.exec_())		#监听退出，如果选择退出，界面就会关掉




