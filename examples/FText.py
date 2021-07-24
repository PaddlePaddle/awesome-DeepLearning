import pandas as pd
import random
import fasttext
import jieba
from sklearn.model_selection import train_test_split

cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5}
"""
函数说明：加载数据
"""
def loadData():


    #利用pandas把数据读进来
    df_technology = pd.read_csv("./data/technology_news.csv",encoding ="utf-8")
    df_technology=df_technology.dropna()    #去空行处理

    df_car = pd.read_csv("./data/car_news.csv",encoding ="utf-8")
    df_car=df_car.dropna()

    df_entertainment = pd.read_csv("./data/entertainment_news.csv",encoding ="utf-8")
    df_entertainment=df_entertainment.dropna()

    df_military = pd.read_csv("./data/military_news.csv",encoding ="utf-8")
    df_military=df_military.dropna()

    df_sports = pd.read_csv("./data/sports_news.csv",encoding ="utf-8")
    df_sports=df_sports.dropna()

    technology=df_technology.content.values.tolist()[1000:21000]
    car=df_car.content.values.tolist()[1000:21000]
    entertainment=df_entertainment.content.values.tolist()[:20000]
    military=df_military.content.values.tolist()[:20000]
    sports=df_sports.content.values.tolist()[:20000]

    return technology,car,entertainment,military,sports

"""
函数说明：停用词
参数说明：
    datapath：停用词路径
返回值：
    stopwords:停用词
"""
def getStopWords(datapath):
    stopwords=pd.read_csv(datapath,index_col=False,quoting=3,sep="\t",names=['stopword'], encoding='utf-8')
    stopwords=stopwords["stopword"].values
    return stopwords

"""
函数说明：去停用词
参数：
    content_line：文本数据
    sentences：存储的数据
    category：文本类别
"""
def preprocess_text(content_line,sentences,category,stopwords):
    for line in content_line:
        try:
            segs=jieba.lcut(line)    #利用结巴分词进行中文分词
            segs=filter(lambda x:len(x)>1,segs)    #去掉长度小于1的词
            segs=filter(lambda x:x not in stopwords,segs)    #去掉停用词
            sentences.append("__lable__"+str(category)+" , "+" ".join(segs))    #把当前的文本和对应的类别拼接起来，组合成fasttext的文本格式
        except Exception as e:
            print (line)
            continue

"""
函数说明：把处理好的写入到文件中，备用
参数说明：
"""
def writeData(sentences,fileName):
    print("writing data to fasttext format...")
    out=open(fileName,'w')
    for sentence in sentences:
        out.write(sentence.encode('utf8')+"\n")
    print("done!")

"""
函数说明：数据处理
"""
def preprocessData(stopwords,saveDataFile):
    technology,car,entertainment,military,sports=loadData()

    #去停用词，生成数据集
    sentences=[]
    preprocess_text(technology,sentences,cate_dic["technology"],stopwords)
    preprocess_text(car,sentences,cate_dic["car"],stopwords)
    preprocess_text(entertainment,sentences,cate_dic["entertainment"],stopwords)
    preprocess_text(military,sentences,cate_dic["military"],stopwords)
    preprocess_text(sports,sentences,cate_dic["sports"],stopwords)

    random.shuffle(sentences)    #做乱序处理，使得同类别的样本不至于扎堆

    writeData(sentences,saveDataFile)

if __name__=="__main__":
    stopwordsFile=r"./data/stopwords.txt"
    stopwords=getStopWords(stopwordsFile)
    saveDataFile=r'train_data.txt'
    preprocessData(stopwords,saveDataFile)
    #fasttext.supervised():有监督的学习
    classifier=fasttext.supervised(saveDataFile,'classifier.model',lable_prefix='__lable__')
    result = classifier.test(saveDataFile)
    print("P@1:",result.precision)    #准确率
    print("R@2:",result.recall)    #召回率
    print("Number of examples:",result.nexamples)    #预测错的例子

    #实际预测
    lable_to_cate={1:'technology'.1:'car',3:'entertainment',4:'military',5:'sports'}

    texts=['中新网 日电 2018 预赛 亚洲区 强赛 中国队 韩国队 较量 比赛 上半场 分钟 主场 作战 中国队 率先 打破 场上 僵局 利用 角球 机会 大宝 前点 攻门 得手 中国队 领先']
    lables=classifier.predict(texts)
    print(lables)
    print(lable_to_cate[int(lables[0][0])])

    #还可以得到类别+概率
    lables=classifier.predict_proba(texts)
    print(lables)

    #还可以得到前k个类别
    lables=classifier.predict(texts，k=3)
    print(lables)

    #还可以得到前k个类别+概率
    lables=classifier.predict_proba(texts，k=3)
    print(lables)