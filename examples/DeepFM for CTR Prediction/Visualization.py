#此段代码不在AI Studio中运行，本地运行，读取上面的训练日志，画图可视化
import matplotlib.pyplot as plt
import xlrd
#plt的字体选择中文四黑
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 打开一个workbook
workbook = xlrd.open_workbook(r'1.xlsx')

# 抓取所有sheet页的名称
worksheets = workbook.sheet_names()
print('worksheets is %s' % worksheets)

# 定位到mySheet
mySheet = workbook.sheet_by_name(u'Sheet1')

# get datas
loss = mySheet.col_values(1)
print(loss)
time = mySheet.col(0)
print('time1',time)
time = [x.value for x in time]
print('time2',time)


#去掉标题行
loss.pop(0)
time.pop(0)

# declare a figure object to plot
fig = plt.figure(1)

# plot loss
plt.plot(time,loss)

plt.title('损失度loss随训练完成度变化曲线')
plt.ylabel('loss')
plt.xticks(range(0,1))
plt.show()