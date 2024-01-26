#!/usr/bin/env python
# coding: utf-8

# 
# ## 1 简介
# 
# 本项目结合长短期记忆（Long short-term memory, LSTM）,以唐诗为数据集通过监督学习的方式，训练生成唐诗。
# 


# ## 2 定义超参数

# In[1]:


from paddle.io import Dataset
import paddle.fluid as fluid
import numpy as np
import paddle
class Config(object):
    num_layers = 3                                      # LSTM层数
    data_path = 'work/tang_poem.npz'                    # 诗歌的文本文件存放路径
    lr = 1e-3                                           # 学习率
    use_gpu = True                                      # 是否使用GPU
    epoch = 20                                  
    batch_size = 4                                      # mini-batch大小
    maxlen = 125                                        # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 1000                                   # 隔batch 可视化一次
    max_gen_len = 200                                   # 生成诗歌最长长度
    model_path = "work/checkpoints/model.params.50"     # 预训练模型路径
    prefix_words = '欲穷千里目，更上一层楼'                 # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words = '老夫聊发少年狂，'                       # 诗歌开始
    model_prefix = 'work/checkpoints/model.params'      # 模型保存路径
    embedding_dim = 256                                 # 词向量维度
    hidden_dim = 512                                    # LSTM hidden层维度


# ## 3.定义DataLoader
# > 数据集由唐诗组成，包含唐诗57580首125字（不足和多余125字的都被补充或者截断）、ix2word以及word2ix共三个字典存储为npz格式

# In[ ]:


paddle.enable_static()
datas = np.load(Config.data_path,allow_pickle=True)
data = datas['data']
#加载映射表
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()

class dataset(Dataset):
    def __init__(self, data):
        super(dataset,self).__init__()
        self.data = data

    def __getitem__(self, idx):
        poem = data[idx]
        return poem

    def __len__(self):
        return len(self.data)

train_dataset = dataset(data)
poem = paddle.static.data(name='poem', shape=[None,125], dtype='float32')
if Config.use_gpu:
    device = paddle.set_device('gpu')
    places = paddle.CUDAPlace(0)
else:
    device = paddle.set_device('cpu')
    places = paddle.CPUPlace()
paddle.disable_static(device)
train_loader = paddle.io.DataLoader(
    train_dataset, 
    places=places, 
    feed_list = [poem],
    batch_size=Config.batch_size, 
    shuffle=True,
    num_workers=2,
    use_buffer_reader=True,
    use_shared_memory=False,
    drop_last=True,
)


# ## 4.定义网络
# 网络由一层Embedding层和三层LSTM层再通过全连接层组成
# > * input:[seq_len,batch_size]
# > * 经过embedding层，embeddings(input)
# >	* output:[batch_size，seq_len,embedding_size]
# > * 经过LSTM，lstm(embeds, (h_0, c_0))，输出output，hidden
# >	* output：[batch, seq_len, hidden_size]
# > * Reshape再进过Linear层判别
# >	* output：[batch*seq_len, vocabsize]

# In[ ]:


import paddle.fluid
class Peom(paddle.nn.Layer):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(Peom, self).__init__()
        self.embeddings = paddle.nn.Embedding(vocab_size,embedding_dim)
        self.lstm = paddle.nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=Config.num_layers,
        )
        self.linear = paddle.nn.Linear(in_features=hidden_dim,out_features=vocab_size)
    def forward(self,input_,hidden=None):
        seq_len, batch_size = paddle.shape(input_)
        embeds = self.embeddings(input_)
        if hidden is None:
            output,hidden = self.lstm(embeds)
        else:
            output,hidden = self.lstm(embeds,hidden)
        output = paddle.reshape(output,[seq_len*batch_size,Config.hidden_dim])
        output = self.linear(output)
        return output,hidden


# ## 5.训练过程
# > * 输入的input为(batch_size,seq_len)
# > * 通过input_,target = data_[:,:-1],data_[:,1:]将每句话分为前n-1个字作为真正的输入，后n-1个字作为label,size都是(batch_size, seq_len-1)
# > * 经过网络，得出output：((seq_len-1)*batch, vocab_size)
# > * 通过label经过reshape将target变成((seq_len-1)*batch)
# 
# 损失函数为:crossEntropy,优化器为:Adam

# In[ ]:


loss_= []
def train():
    model = Peom(
        len(word2ix),
        embedding_dim = Config.embedding_dim,
        hidden_dim = Config.hidden_dim
    )
    # state_dict = paddle.load(Config.model_path)
    # model.set_state_dict(state_dict)
    optim = paddle.optimizer.Adam(parameters=model.parameters(),learning_rate=Config.lr)
    lossf = paddle.nn.CrossEntropyLoss()
    for epoch in range(Config.epoch):
        for li,data in enumerate(train_loader()):
            optim.clear_grad()
            data = data[0]
            #data = paddle.transpose(data,(1,0))
            x = paddle.to_tensor(data[:,:-1])
            y = paddle.to_tensor(data[:,1:],dtype='int64')
            y = paddle.reshape(y,[-1])
            y = paddle.to_tensor(y,dtype='int64')
            output,hidden = model(x)
            loss = lossf(output,y)
            loss.backward()
            optim.step()
            loss_.append(loss.numpy()[0])

            if li % Config.plot_every == 0:
                print('Epoch ID={0}\t Batch ID={1}\t Loss={2}'.format(epoch, li, loss.numpy()[0]))
                
                results = list(Config.start_words)
                start_words_len = len(Config.start_words)
                # 第一个词语是<START>
                input = paddle.to_tensor([word2ix['<START>']])
                input = paddle.reshape(input,[1,1])
                hidden = None

                # 若有风格前缀，则先用风格前缀生成hidden
                if Config.prefix_words:
                    # 第一个input是<START>，后面就是prefix中的汉字
                    # 第一个hidden是None，后面就是前面生成的hidden
                    for word in Config.prefix_words:
                        output, hidden = model(input, hidden)
                        input = paddle.to_tensor([word2ix[word]])
                        input = paddle.reshape(input,[1,1])

                # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
                # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
                for i in range(Config.max_gen_len):
                    output, hidden = model(input, hidden)
                    # print(output.shape)
                    # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
                    # 最后的hidden
                    if i < start_words_len:
                        w = results[i]
                        input = paddle.to_tensor([word2ix[w]])
                        input = paddle.reshape(input,[1,1])
                    # 否则将output作为下一个input进行
                    else:
                        # print(output.data[0].topk(1))
                        _,top_index = paddle.fluid.layers.topk(output[0],k=1)
                        top_index = top_index.numpy()[0]
                        w = ix2word[top_index]
                        results.append(w)
                        input = paddle.to_tensor([top_index])
                        input = paddle.reshape(input,[1,1])
                    if w == '<EOP>':
                        del results[-1]
                        break
                results = ''.join(results)
                print(results)
        paddle.save(model.state_dict(), Config.model_prefix)
train()



# ## 6.Loss变化过程

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(15, 6))
x = np.arange(len(loss_))
plt.title('Loss During Training')
plt.xlabel('Number of Batch')
plt.plot(x,np.array(loss_))
plt.savefig('work/Loss During Training.png')
plt.show()


# ## 7.生成唐诗
# ### 7.1 模式一 <首句续写唐诗>
# 例如：”老夫聊发少年狂“
# > 老夫聊发少年狂，嫁得双鬟梳似桃。</br>	
# > 青丝长发娇且红，輭舞脸低时未昏。</br>	
# > 娇嚬欲尽一双转，笑语千里相竞言。</br>		       	
# > 朝阳上去花欲尽，花时且落花前过。</br>	
# > 秾雨霏霏满地晓，红妆白鸟飞下郭。</br>	
# > 灯前织女嫁新租，袖里垂纶舞袖舞。</br>	
# > 罗袖焰扬簷下樱，一宿十二花绵绵。</br>	
# > 西施夹道春风暖，嫩粉萦丝弄金蘂。</br>	  	
# ### 7.2 模式二<藏头诗>
# 例如：”夜月一帘幽梦春风十里柔情“
# > 夜半星初洽，月明星未稀。</br>	
# > 一缄琼烛动，帘外玉环飞。</br>	
# > 幽匣光华溢，梦中形影微。</br>	
# > 春风吹蕙笏，风绪拂莓苔。</br>	
# > 十月涵金井，里尘氛祲微。</br>	
# > 柔荑暎肌骨，情酒围唇肌。</br>		

# In[ ]:


# 给定首句生成诗歌
def generate(model, start_words, prefix_words=None):
    results = list(start_words)
    start_words_len = len(start_words)
    # 第一个词语是<START>
    input = paddle.to_tensor([word2ix['<START>']])
    input = paddle.reshape(input,[1,1])
    hidden = None

    # 若有风格前缀，则先用风格前缀生成hidden
    if prefix_words:
        # 第一个input是<START>，后面就是prefix中的汉字
        # 第一个hidden是None，后面就是前面生成的hidden
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = paddle.to_tensor([word2ix[word]])
            input = paddle.reshape(input,[1,1])

    # 开始真正生成诗句，如果没有使用风格前缀，则hidden = None，input = <START>
    # 否则，input就是风格前缀的最后一个词语，hidden也是生成出来的
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        # print(output.shape)
         # 如果还在诗句内部，输入就是诗句的字，不取出结果，只为了得到
        # 最后的hidden
        if i < start_words_len:
            w = results[i]
            input = paddle.to_tensor([word2ix[w]])
            input = paddle.reshape(input,[1,1])
        # 否则将output作为下一个input进行
        else:
            # print(output.data[0].topk(1))
            _,top_index = paddle.fluid.layers.topk(output[0],k=1)
            top_index = top_index.numpy()[0]
            w = ix2word[top_index]
            results.append(w)
            input = paddle.to_tensor([top_index])
            input = paddle.reshape(input,[1,1])
        if w == '<EOP>':
            del results[-1]
            break
    results = ''.join(results)
    return results


# 生成藏头诗
def gen_acrostic(model, start_words, prefix_words=None):
    result = []
    start_words_len = len(start_words)
    input = paddle.to_tensor([word2ix['<START>']])
    input = paddle.reshape(input,[1,1])
    # 指示已经生成了几句藏头诗
    index = 0
    pre_word = '<START>'
    hidden = None

    # 存在风格前缀，则生成hidden
    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = paddle.to_tensor([word2ix[word]])
            input = paddle.reshape(input,[1,1])

    # 开始生成诗句
    for i in range(Config.max_gen_len):
        output, hidden = model(input, hidden)
        _,top_index = paddle.fluid.layers.topk(output[0],k=1)
        top_index = top_index.numpy()[0]
        w = ix2word[top_index]
        # 说明上个字是句末
        if pre_word in {'。', '，', '?', '！', '<START>'}:
            if index == start_words_len:
                break
            else:
                w = start_words[index]
                index += 1
                # print(w,word2ix[w])
                input = paddle.to_tensor([word2ix[w]])
                input = paddle.reshape(input,[1,1])
        else:
            input = paddle.to_tensor([top_index])
            input = paddle.reshape(input,[1,1])
        result.append(w)
        pre_word = w
    result = ''.join(result)
    return result

#读取训练好的模型
model = Peom(
        len(word2ix),
        embedding_dim = Config.embedding_dim,
        hidden_dim = Config.hidden_dim
    )
state_dict = paddle.load('work/checkpoints/model.params.50')
model.set_state_dict(state_dict)
print('[*]模式一：首句续写唐诗：')
#生成首句续写唐诗（prefix_words为生成意境的诗句）
print(generate(model, start_words="春江潮水连海平，", prefix_words="滚滚长江东逝水，浪花淘尽英雄。"))
print('[*]模式二：藏头诗：')
#生成藏头诗
print(gen_acrostic(model, start_words="夜月一帘幽梦春风十里柔情", prefix_words="落花人独立，微雨燕双飞。"))

