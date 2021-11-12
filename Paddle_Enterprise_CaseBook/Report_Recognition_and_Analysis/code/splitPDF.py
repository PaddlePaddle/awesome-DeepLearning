# pdf切分
import os, fitz, time

def pdf2png(pdfPath, imgPath, zoom_x=2, zoom_y=2, rotation_angle=0):
    '''
    # 将PDF转化为图片
    pdfPath pdf文件的路径
    imgPath 图像要保存的文件夹
    zoom_x x方向的缩放系数
    zoom_y y方向的缩放系数
    rotation_angle 旋转角度
    互联网行业《先进无线技术应用情况调研》全球版：借助5G和Wi-Fi6加速企业创新和转型 6.png
    zoom_x=5 2977x4210 885kb 0.256
    zoom_x=2 1191x1684 290kb 0.06s 4800s
    '''

    time_start = time.time()
    # 打开PDF文件
    pdf = fitz.open(pdfPath)
    # 逐页读取PDF
    for pg in range(0, pdf.pageCount):
        page = pdf[pg]
        # 设置缩放和旋转系数
        trans = fitz.Matrix(zoom_x, zoom_y) #.preRotate(rotation_angle)
        pm = page.getPixmap(matrix=trans, alpha=False)

        if pm.width>2000 or pm.height>2000:
            pm = page.getPixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        pm.writePNG(imgPath + str(pg) + ".jpeg")
    pdf.close()
    time_end = time.time()
    time_cost = time_end - time_start
    print('totally cost: {}, page: {}, each page cost: {}'.format(time_cost, pg+1, time_cost/(pg+1)))

if __name__ == '__main__':
    pdfFolder = 'ResearchReport'
    for p in os.listdir(pdfFolder):
        pdfPath = pdfFolder+'/'+p
        imgPath = pdfFolder+'/'+os.path.basename(p)[:-4]+'/'
        print(imgPath)
        os.mkdir(imgPath)
        pdf2png(pdfPath, imgPath)