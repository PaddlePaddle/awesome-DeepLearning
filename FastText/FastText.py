import jieba
import os

basedir = "/home/python_home/WeiZhongChuang/ML/Embedding/THUCNews/" #这是我的文件地址，需跟据文件夹位置进行更改
dir_list = ['affairs','constellation','economic','edu','ent','fashion','game','home','house','lottery','science','sports','society','stock']
##生成fastext的训练和测试数据集

ftrain = open("news_fasttext_train.txt","w")
ftest = open("news_fasttext_test.txt","w")

num = -1
for e in dir_list:
    num += 1
    indir = basedir + e + '/'
    files = os.listdir(indir)
    count = 0
    for fileName in files:
        count += 1            
        filepath = indir + fileName
        with open(filepath,'r') as fr:
            text = fr.read()
        text = str(text.encode("utf-8"),'utf-8')
        seg_text = jieba.cut(text.replace("\t"," ").replace("\n"," "))
        outline = " ".join(seg_text)
        outline = outline + "\t__label__" + e + "\n"
#         print outline
#         break

        if count < 10000:
            ftrain.write(outline)
            ftrain.flush()
            continue
        elif count  < 20000:
            ftest.write(outline)
            ftest.flush()
            continue
        else:
            break

ftrain.close()
ftest.close()
print('完成输出数据！')


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import fasttext
#训练模型
classifier = fasttext.train_supervised("news_fasttext_train.txt",label_prefix="__label__")

#load训练好的模型
#classifier = fasttext.load_model('news_fasttext.model.bin', label_prefix='__label__')
print('训练完成！')

result = classifier.test("news_fasttext_test.txt")
print('precision：', result[1])

labels_right = []
texts = []
with open("news_fasttext_test.txt") as fr:
    for line in fr:
        line = str(line.encode("utf-8"), 'utf-8').rstrip()
        labels_right.append(line.split("\t")[1].replace("__label__",""))
        texts.append(line.split("\t")[0])
    #     print labels
    #     print texts
#     break
labels_predict = [term[0] for term in classifier.predict(texts)[0]] #预测输出结果为二维形式
# print labels_predict

text_labels = list(set(labels_right))
text_predict_labels = list(set(labels_predict))
print(text_predict_labels)
print(text_labels)
print()

A = dict.fromkeys(text_labels,0)  #预测正确的各个类的数目
B = dict.fromkeys(text_labels,0)   #测试数据集中各个类的数目
C = dict.fromkeys(text_predict_labels,0) #预测结果中各个类的数目
for i in range(0,len(labels_right)):
    B[labels_right[i]] += 1
    C[labels_predict[i]] += 1
    if labels_right[i] == labels_predict[i].replace('__label__', ''):
        A[labels_right[i]] += 1

print('预测正确的各个类的数目:', A) 
print()
print('测试数据集中各个类的数目:', B)
print()
print('预测结果中各个类的数目:', C)
print()
#计算准确率，召回率，F值
for key in B:
    try:
        r = float(A[key]) / float(B[key])
        p = float(A[key]) / float(C['__label__' + key])
        f = p * r * 2 / (p + r)
        print("%s:\t p:%f\t r:%f\t f:%f" % (key,p,r,f))
    except:
        print("error:", key, "right:", A.get(key,0), "real:", B.get(key,0), "predict:",C.get(key,0))