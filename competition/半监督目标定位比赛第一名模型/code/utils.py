import os
import glob


# 输入标签集的地址 win or linux 可能有点路径问题 就是\\ 与 /的差别，这里是基于win，如果路径有问题需要换成linux的/
def write_file_name(path):
    text  = glob.glob(path + '\*\*')
    if '\\' in text[0]:
        text  = [i.split('\\')[-1] for i in text]
    else:
        text = [i.split('/')[-1] for i in text]
    cwd   = os.path.abspath(os.path.dirname(path))
    with open(cwd + '\\train.txt', 'a+') as f:
        for i in text:
            f.write(i[0:-4]+'\n')


write_file_name(r'I:\compa\pix\mask')


