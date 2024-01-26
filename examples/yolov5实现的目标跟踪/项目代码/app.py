from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QInputDialog, QGridLayout, QLabel, QPushButton, QFrame, \
    QFileDialog
from track import detect
import sys


class InputDialog(QWidget):
    def __init__(self):
        super(InputDialog, self).__init__()
        self.nameLabel = QLabel("MOT挑战-可视.mp4")
        self.initUi()
        self.button = QPushButton('分析', self)
        self.button.clicked.connect(self.analysisMovie)
        self.button.move(200, 250)

    def initUi(self):
        self.setWindowTitle("目标跟踪")
        self.setGeometry(800, 800, 500, 300)

        label1 = QLabel("视频路径:")

        self.nameLabel.setFrameStyle(QFrame.Panel | QFrame.Sunken)

        nameButton = QPushButton("修改")
        nameButton.clicked.connect(self.selectFile)

        mainLayout = QGridLayout()
        mainLayout.addWidget(label1, 0, 0)
        mainLayout.addWidget(self.nameLabel, 0, 1)
        mainLayout.addWidget(nameButton, 0, 2)

        self.setLayout(mainLayout)

    def selectFile(self):
        filename = QFileDialog.getOpenFileName(self, 'Open file', '.')
        print(type(filename))
        self.nameLabel.setText(filename[0])

    def analysisMovie(self):
        detect(self.nameLabel.text())
        print('正在对' + self.nameLabel.text() + '进行分析...')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    showWindow = InputDialog()
    showWindow.show()
    sys.exit(app.exec_())
