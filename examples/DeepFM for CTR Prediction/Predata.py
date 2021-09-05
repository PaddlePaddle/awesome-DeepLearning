# 查看数据格式
# !cd slot_train_data_full/ && ls -lha && head part-0
# !cd slot_test_data_full/ && ls -lha && head part-220
import os
import subprocess
def wc_count(file_name):
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])
def wc_count_dir(dirPath):#统计最终一共有多少行数据
    cnt=0
    fileList=os.listdir(dirPath)
    for fileName in fileList:
        cnt+=wc_count(dirPath.strip('/')+'/'+fileName)
    return cnt


def predata(rawLine):
    '''
    数据处理，缺失值填充，原始数据拆分
    '''
    # 划分特征
    padding = '0'
    fea_vals = rawLine.strip().split(' ')
    label = ['click']
    sparse_fea = label + [str(x) for x in range(1, 27)]  # 提取所有的特征
    dense_fea = ['dense_inputs']
    dense_fea_dim = 13  # 稠密特征维度为13
    slots = 1 + 26 + 13
    slots_fea = sparse_fea + dense_fea
    output = {}
    for fea_val in fea_vals:
        fea_val = fea_val.split(':')  # 根据数据格式，按":"划分
        if len(fea_val) == 2:
            fea, val = fea_val
        else:
            continue
        if fea not in output.keys():
            output[fea] = [val]  # 连续特征缺失，添加新特征
        else:
            output[fea].append(val)  # 末尾添加

    # 填充
    if len(output.keys()) != slots:
        for fea in slots_fea:
            if fea in sparse_fea:  # 稀疏特征
                if fea not in output.keys(): output[fea] = [padding]
            elif fea not in output.keys():  # 连续特征完全缺失
                output[fea] = [padding] * dense_fea_dim  # 连续特征部分缺失
            elif len(output[fea]) < dense_fea_dim:
                output[fea].extend([padding] * (dense_fea_dim - len(output[fea])))
    data = []
    for fea in slots_fea: data.extend(output[fea])
    return data