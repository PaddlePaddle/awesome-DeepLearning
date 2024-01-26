from PIL import Image
import numpy as np
from frcnn import FRCNN

import os
import xlsxwriter
from time import time


workbook = xlsxwriter.Workbook('model_data/result.xlsx')  #建立文件
worksheet = workbook.add_worksheet('result')

frcnn = FRCNN()

imgs = os.listdir('img/first_examples')
imgs.sort(key=lambda x:int(x[2:-5]))

row = 2
clo = 0
length = 0

av_ac = 0
kind_num = {'haomao':0,
                 'furongwang':0,
                 'zuanshi':0,
                 'zhonghua':0,
                 'changan':0,
                 'yunyan':0,
                 'nanjing':0,
                 'liqun':0,
                 'houwang':0,
                 'lanzhou':0}
av_time = 0
all_num = 0
while True:
    for i in range(len(imgs)):
        print(i)
        worksheet.write(row - 1, clo, '图片编号%s' %imgs[i][:-4])
        for m in range(1, 11):
            worksheet.write(row - 1, clo + m, '类别%d' % (m))
        worksheet.write(row, clo, '类别名称')
        worksheet.write(row + 1, clo, '数量')
        worksheet.write(row + 2, clo, '检测精度')
        worksheet.write(row + 3, clo, '检测速度/s')
        worksheet.write(row + 4, clo,'中心点x坐标')
        worksheet.write(row + 5, clo,'中心点y坐标')

        img = 'img/first_examples/' + imgs[i]
        image = Image.open(img)
        t1 = time()
        r_image, center_x, center_y, labels, numbers = frcnn.detect_image(image)
        t2 = time()
        number = list(numbers.values())
        location = np.nonzero(number)[0]
        kinds = []
        for n in numbers:
            kinds.append(n)
        for j in range(len(location)):
            label1 = list(str(labels[j]))
            label1.pop(0)
            label1.pop(0)
            label1.pop()
            k = label1.index(' ')
            kind = ''.join(label1[:k])
            acuracy = float(''.join(label1[k+1:]))

            av_ac = av_ac + acuracy
            av_time = av_time + (t2-t1)
            kind_num[kind] = kind_num[kind] + 1

            print('kind = %s\nnumber = %d\nacuracy = %.2f\ncenter_x = %.2f\ncenter_y = %.2f\n'
                  % (kinds[location[j]], number[location[j]], acuracy, center_x[j], center_y[j]))
            worksheet.write(row, clo + location[j] + 1, kinds[location[j]])
            worksheet.write(row + 1, clo + location[j] + 1, number[location[j]])
            worksheet.write(row + 2, clo + location[j] + 1, float(acuracy))
            worksheet.write(row + 3, clo + location[j] + 1, t2 - t1)
            worksheet.write(row + 4, clo + location[j] + 1, center_x[j])
            worksheet.write(row + 5, clo + location[j] + 1, center_y[j])

        row += 9
        '''r_image,_,_,_,_ = frcnn.detect_image(image)
        r_image.show()'''
    break

for values in kind_num.values():
    all_num = values + all_num
av_ac = av_ac / all_num
av_time = av_time / all_num

kind_name = []
for i in kind_num:
    kind_name.append(i)

worksheet.write(row - 1, clo, '所有图片')
for m in range(1, 11):
    worksheet.write(row - 1, clo + m, '类别%d' % (m))
    worksheet.write(row, clo + m, kind_name[m-1])
worksheet.write(row, clo, '类别名称')
worksheet.write(row + 1, clo, '总数量')
worksheet.write(row + 2, clo, '平均检测精度')
worksheet.write(row + 3, clo, '平局检测速度/s')



kinds_num = list(kind_num.values())
for m in range(1,11):
    worksheet.write(row + 1, clo + m, kinds_num[m-1])
worksheet.write(row + 2, clo +1, av_ac)
worksheet.write(row + 3, clo +1, av_time)

workbook.close()
