import cv2
import imutils
from tkinter import Tk, Button, Frame, Label, NW
from tkinter import filedialog
from PIL import Image,ImageTk
from AIDetector_pytorch import Detector

class UI():
    def __init__(self):
        self.go_on = True
        self.dir = 'input.mp4'

        self.root = Tk()
        self.frm1 = Frame(self.root)
        self.frm2 = Frame(self.root)
        self.lab1 = Label(self.frm1)
        self.butt1 = Button(self.frm2, text='视频目标检测')
        self.butt2 = Button(self.frm2, text='实时目标检测')
        self.butt3 = Button(self.frm2, text='取消检测')

        self.createpage()

    def createpage(self):
        self.root.title("视频图像目标检测")
        self.root.geometry(("1080x800"))
        self.frm1.config(width=1080, height=720)
        self.frm2.config(width=1080, height=80)
        self.frm1.place(x=0, y=0)
        self.frm2.place(x=0, y=720)
        self.lab1.place(in_=self.frm1, anchor=NW)
        self.butt1.config(
                    bd=1,
                    width=10,
                    height=2,
                    command=lambda: self.demo(filedialog.askopenfilename())
                    )
        self.butt1.place(in_=self.frm2, x=300, y=0)  # 点击按钮执行main函数
        self.butt2.config(
                    bd=1,
                    width=10,
                    height=2,
                    command=lambda: self.demo(0)
                    )
        self.butt2.place(in_=self.frm2, x=500, y=0)
        self.butt3.config(
                    bd=1,
                    width=10,
                    height=2,
                    command=self.eait_demo
                    )
        self.butt3.place(in_=self.frm2, x=700, y=0)

        self.root.mainloop()

    def demo(self, dir='input.mp4'):

        func_status = {}
        func_status['headpose'] = None
        self.go_on = True

        det = Detector()
        cap = cv2.VideoCapture(dir)                 # 打开视频文件
        fps = int(cap.get(5))                       # 读取帧率
        print('fps:', fps)
        t = int(1000 / fps)
        size = None
        videoWriter = None


        while True:

            _, im = cap.read()
            if im is None:
                break

            result = det.feedCap(im, func_status)
            result = result['frame']
            result = imutils.resize(result, width=1080)
            # height, width, _ = result.shape
            if videoWriter is None:
                fourcc = cv2.VideoWriter_fourcc(
                    'm', 'p', '4', 'v')  # opencv3.0
                videoWriter = cv2.VideoWriter(
                    'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))      # 初始化结果
            videoWriter.write(result)           # 写入结果

            img_open = Image.fromarray(result)
            img = ImageTk.PhotoImage(img_open)
            self.lab1.update()
            self.lab1.config(image=img)
            self.lab1.image = img                   # keep a reference

            if self.go_on is False:
                break

        cap.release()
        videoWriter.release()

    def eait_demo(self):
        self.go_on = False

if __name__ == '__main__':
    UI()



