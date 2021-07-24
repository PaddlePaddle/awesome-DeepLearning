```python
# 查看当前挂载的数据集目录, 该目录下的变更重启环境后会自动还原
# View dataset directory. 
# This directory will be recovered automatically after resetting environment. 
!ls /home/aistudio/data
```


```python
# 查看工作区文件, 该目录下的变更将会持久保存. 请及时清理不必要的文件, 避免加载过慢.
# View personal work directory. 
# All changes under this directory will be kept even after reset. 
# Please clean unnecessary files in time to speed up environment loading. 
!ls /home/aistudio/work
```


```python
# 如果需要进行持久化安装, 需要使用持久化路径, 如下方代码示例:
# If a persistence installation is required, 
# you need to use the persistence path as the following: 
!mkdir /home/aistudio/external-libraries
!pip install beautifulsoup4 -t /home/aistudio/external-libraries
```

    mkdir: cannot create directory ‘/home/aistudio/external-libraries’: File exists
    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting beautifulsoup4
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/d1/41/e6495bd7d3781cee623ce23ea6ac73282a373088fcd0ddc809a047b18eae/beautifulsoup4-4.9.3-py3-none-any.whl (115kB)
    [K     |████████████████████████████████| 122kB 14.9MB/s eta 0:00:01
    [?25hCollecting soupsieve>1.2; python_version >= "3.0" (from beautifulsoup4)
      Downloading https://mirror.baidu.com/pypi/packages/36/69/d82d04022f02733bf9a72bc3b96332d360c0c5307096d76f6bb7489f7e57/soupsieve-2.2.1-py3-none-any.whl
    Installing collected packages: soupsieve, beautifulsoup4
    Successfully installed beautifulsoup4-4.9.3 soupsieve-2.2.1
    [33mWARNING: Target directory /home/aistudio/external-libraries/beautifulsoup4-4.9.3.dist-info already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/bs4 already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve-2.2.1.dist-info already exists. Specify --upgrade to force replacement.[0m
    [33mWARNING: Target directory /home/aistudio/external-libraries/soupsieve already exists. Specify --upgrade to force replacement.[0m



```python
# 同时添加如下代码, 这样每次环境(kernel)启动的时候只要运行下方代码即可: 
# Also add the following code, 
# so that every time the environment (kernel) starts, 
# just run the following code: 
import sys 
sys.path.append('/home/aistudio/external-libraries')
```

请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 

CBOW（Continuous Bag of Words）
tf-idf算法是通过一种统计学的方式来用文章中的词的重要程度，转化成向量来表示一篇文章的。速度较快，但是准确率不如深度学习的方法

CBOW就是挑一个要预测的词，来学习这个词前后文中词语和预测词的关系，那每个词都可以在一个空间中表示出来，可以通过空间位置知道词语之间的对应关系，理论上，语义越相近的词语将会距离更近

corpus = [
    # numbers
    "5 2 4 8 6 2 3 6 4",
    "4 8 5 6 9 5 5 6",
    "1 1 5 2 3 3 8",
    "3 6 9 6 8 7 4 6 3",
    "8 9 9 6 1 4 3 4",
    "1 0 2 0 2 1 3 3 3 3 3",
    "9 3 3 0 1 4 7 8",
    "9 9 8 5 6 7 1 2 3 0 1 0",

    # alphabets, expecting that 9 is close to letters
    "a t g q e h 9 u f",
    "e q y u o i p s",
    "q o 9 p l k j o k k o p",
    "h g y i u t t a e q",
    "i k d q r e 9 e a d",
    "o p d g 9 s a f g a",
    "i u y g h k l a s w",
    "o l u y a o g f s",
    "o p i u y g d a s j d l",
    "u k i l o 9 l j s",
    "y g i s h k j l f r f",
    "i o h n 9 9 d 9 f a 9",
]
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
例如这里又字母也有数字，假设他们都是单词，那根据前后文关系，其实应该是学到字母们在向量空间上的位置更相近，而数字的位置更相近，对于数字9这个与字母也前后文关系密切，与数字也前后文关系密切的，应该空间上离两者都很近

构造模型的训练集，这里是将一句话中滑动移动5个单词的窗口，以前后两个，共四个单词作为训练数据的特征，中间那个词作为训练数据的标签

# 3.定义产生训练数据的方法
class Dataset:
    def __init__(self, x, y, v2i, i2v):
        self.x, self.y = x, y
        self.v2i, self.i2v = v2i, i2v
        self.vocab = v2i.keys()

    def sample(self, n):
        b_idx = np.random.randint(0, len(self.x), n) # 产生随机数
        bx, by = self.x[b_idx], self.y[b_idx] # 使顺序打散训练，这样每一次调用，都会产生shuffle的数据
        return bx, by

    @property
    def num_word(self):
        return len(self.v2i)


def process_w2v_data(corpus, skip_window=2, method="skip_gram"):
    all_words = [sentence.split(" ") for sentence in corpus]
    all_words = np.array(list(itertools.chain(*all_words))) # 连成一条
    # vocab sort by decreasing frequency for the negative sampling below (nce_loss).
    vocab, v_count = np.unique(all_words, return_counts=True) # 统计有多少种不同的词，以及个数
    vocab = vocab[np.argsort(v_count)[::-1]] # 按照个数从多到少排序

    print("all vocabularies sorted from more frequent to less frequent:\n", vocab)
    v2i = {v: i for i, v in enumerate(vocab)} # 单词即对应的索引，从大到小拍多少号
    i2v = {i: v for v, i in v2i.items()}  # 从索引找到单词

    # pair data
    pairs = []
    js = [i for i in range(-skip_window, skip_window + 1) if i != 0]
    # -2到2一共五个单词，去掉最中间的，窗口大小为5

    for c in corpus:
        words = c.split(" ")
        w_idx = [v2i[w] for w in words] # 把每一行文本的单词都换成索引放在列表里
        if method == "skip_gram":
            for i in range(len(w_idx)):
                for j in js:
                    if i + j < 0 or i + j >= len(w_idx):
                        continue
                    pairs.append((w_idx[i], w_idx[i + j]))  # (center, context) or (feature, target)
        elif method.lower() == "cbow":
            for i in range(skip_window, len(w_idx) - skip_window): # 在这一行列表上开始滑动窗口，每一次华东都包含5个单词
                context = [] # 装每个窗口的五个单词的索引，context里面装的是前后共四个
                for j in js:
                    context.append(w_idx[i + j])
                pairs.append(context + [w_idx[i]])  # (contexts, center) or (feature, target)
        else:
            raise ValueError
    pairs = np.array(pairs)
    print("5 example pairs:\n", pairs[:5])
    if method.lower() == "skip_gram":
        x, y = pairs[:, 0], pairs[:, 1]
    elif method.lower() == "cbow":
        x, y = pairs[:, :-1], pairs[:, -1]
    else:
        raise ValueError
    return Dataset(x, y, v2i, i2v)
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
首先是把所有的单词进行一个词频的统计以及排序，为他们安排一个索引
![](https://ai-studio-static-online.cdn.bcebos.com/fa891b99dd0243ca813a42de15f2534a6100ad776cec475c9722c25335f52209)
所有的单词排序从出现词频最多的到最低的，同时每个词具有了一个索引，把上述的整段文本，用索引来转化
![](https://ai-studio-static-online.cdn.bcebos.com/5ff42d6ca5e9449c9ee7ff2e1c2884921c8b9f0f053d4ab89af470ca393dfaab)
然后对每一句话开始以窗口为5划窗，中间的摘出来作为标签，那么特征就是：
[16, 14, 12, 3]
[14, 9, 3, 14]
[9, 12, 14, 1]
[12, 3, 1, 3]
[3, 14, 3, 9]
[9, 12, 3, 0]
[12, 16, 0, 16]
[16, 3, 16, 16]
[3, 0, 16, 3]
[5, 5, 14, 1]
[5, 16, 1, 1]
[16, 14, 1, 12]
[1, 3, 3, 12]
[3, 0, 12, 25]
[0, 3, 25, 9]
[3, 12, 9, 3]
[12, 25, 3, 1]
[12, 0, 3, 5]
[0, 0, 5, 9]
[0, 3, 9, 1]
[3, 5, 1, 9]
[5, 23, 23, 14]
[23, 14, 14, 5]
[14, 23, 5, 1]
[23, 14, 1, 1]
[14, 5, 1, 1]
[5, 1, 1, 1]
[1, 1, 1, 1]
[0, 1, 23, 5]
[1, 1, 5, 9]
[1, 23, 9, 25]
[23, 5, 25, 12]
[0, 0, 16, 3]
[0, 12, 3, 25]
[12, 16, 25, 5]
[16, 3, 5, 14]
[3, 25, 14, 1]
[25, 5, 1, 23]
[5, 14, 23, 5]
[14, 1, 5, 23]
[4, 26, 22, 19]
[26, 7, 19, 20]
[7, 22, 20, 0]
[22, 19, 0, 13]
[19, 20, 13, 18]
[19, 22, 13, 2]
[22, 17, 2, 6]
[17, 13, 6, 21]
[13, 2, 21, 8]
[22, 2, 21, 10]
[2, 0, 10, 11]
[0, 21, 11, 24]
[21, 10, 24, 2]
[10, 11, 2, 11]
[11, 24, 11, 11]
[24, 2, 11, 2]
[2, 11, 2, 21]
[20, 7, 6, 13]
[7, 17, 13, 26]
[17, 6, 26, 26]
[6, 13, 26, 4]
[13, 26, 4, 19]
[26, 26, 19, 22]
[6, 11, 22, 27]
[11, 15, 27, 19]
[15, 22, 19, 0]
[22, 27, 0, 19]
[27, 19, 19, 4]
[19, 0, 4, 15]
[2, 21, 7, 0]
[21, 15, 0, 8]
[15, 7, 8, 4]
[7, 0, 4, 18]
[0, 8, 18, 7]
[8, 4, 7, 4]
[6, 13, 7, 20]
[13, 17, 20, 11]
[17, 7, 11, 10]
[7, 20, 10, 4]
[20, 11, 4, 8]
[11, 10, 8, 28]
[2, 10, 17, 4]
[10, 13, 4, 2]
[13, 17, 2, 7]
[17, 4, 7, 18]
[4, 2, 18, 8]
[2, 21, 13, 17]
[21, 6, 17, 7]
[6, 13, 7, 15]
[13, 17, 15, 4]
[17, 7, 4, 8]
[7, 15, 8, 24]
[15, 4, 24, 15]
[4, 8, 15, 10]
[13, 11, 10, 2]
[11, 6, 2, 0]
[6, 10, 0, 10]
[10, 2, 10, 24]
[2, 0, 24, 8]
[17, 7, 8, 20]
[7, 6, 20, 11]
[6, 8, 11, 24]
[8, 20, 24, 10]
[20, 11, 10, 18]
[11, 24, 18, 27]
[24, 10, 27, 18]
[6, 2, 29, 0]
[2, 20, 0, 0]
[20, 29, 0, 15]
[29, 0, 15, 0]
[0, 0, 0, 18]
[0, 15, 18, 4]
[15, 0, 4, 0]

在每个list后面加上标签构成pairs，pairs里面就是若干5维的列表：
![](https://ai-studio-static-online.cdn.bcebos.com/675e167d25f549ee876e1564f19afdad916395ac934645188916744a8622b20c)

