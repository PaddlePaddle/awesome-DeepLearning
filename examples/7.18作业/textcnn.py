import torch
import torch.nn as nn
import torch.nn.functional as F
 
sentence = [['The movie is great'],['Tom has a headache today'],['I think the apple is bad'],\
            ['You are so beautiful']]
Label = [1,0,0,1]
 
test_sentence = ['The apple is great']
 
class sentence2id:
    def __init__(self,sentence):
        self.sentence = sentence
        self.dic = {}
        self.words = []
        self.words_num = 0
    
    def sen2sen(self,sentence): ##大写转小写
        senten = []
        if type(sentence[0])== list:
            for sen in sentence:
                sen = sen[0].lower()
                senten.append([sen])           
        else:
            senten.append(sentence)
            senten = self.sen2sen(senten)
        return senten
         
            
    def countword(self): ##统计单词个数 
        ############[建库过程不涉及到test模块]#############
        for sen in self.sentence:
            sen = sen[0].split(' ') ##空格分隔
            for word in sen:
                self.words.append(word.lower())
        self.words = list(set(self.words))
        self.words = sorted(self.words)
        self.words_num = len(self.words)
        return self.words,self.words_num
    
    def word2id(self): ### 创建词汇表
        flag = 1
        for word in self.words:
            if flag <= self.words_num:
                self.dic[word] = flag
                flag += 1
        #print(self.dic)
        return self.dic
    
    def sen2id(self,sentence): ###
        sentence = self.sen2sen(sentence)
        sentoid = []
        for sen in sentence:
            senten = []
            for word in sen[0].split():
                senten.append(self.dic[word])
            sentoid.append(senten)
        return sentoid
                
def padded(sentence,pad_token): #token'<pad>'
    max_len = len(sentence[0])
    for i in range(0,len(sentence)-1):
        if max_len < len(sentence[i+1]):
            max_len = len(sentence[i+1])
        i += 1
    for i in range(0,len(sentence)):
        for j in range(0,max_len-len(sentence[i])):
            sentence[i].append(pad_token)
    return sentence
 
 
class ModelEmbeddings(nn.Module):
    def __init__(self,words_num,embed_size,pad_token): 
        super(ModelEmbeddings, self).__init__()
        self.words_num = words_num
        self.embed_size = embed_size
        self.Embedding = nn.Embedding(words_num,embed_size,pad_token)
 
class textCNN(nn.Module):
    def __init__(self,words_num,embed_size,class_num,dropout_rate=0.1):
        super(textCNN, self).__init__()
        self.words_num = words_num
        self.embed_size = embed_size 
        self.class_num = class_num
        
        self.conv1 = nn.Conv2d(1,3,(2,self.embed_size)) ###in_channels, out_channels, kernel_size
        self.conv2 = nn.Conv2d(1,3,(3,self.embed_size)) 
        self.conv3 = nn.Conv2d(1,3,(4,self.embed_size))
        
        self.max_pool1 = nn.MaxPool1d(5)
        self.max_pool2 = nn.MaxPool1d(4)
        self.max_pool3 = nn.MaxPool1d(3)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(3*3*1,class_num)
        # 3 -> out_channels 3 ->kernel_size 1 ->max_pool
    
    def forward(self,sen_embed): #(batch,max_len,embed_size)
        sen_embed = sen_embed.unsqueeze(1) #(batch,in_channels,max_len,embed_size)
        
        conv1 = F.relu(self.conv1(sen_embed))  # ->(batch_size,out_channels.size,1)
        conv2 = F.relu(self.conv2(sen_embed))
        conv3 = F.relu(self.conv3(sen_embed))
    
        conv1 = torch.squeeze(conv1,dim=3)
        conv2 = torch.squeeze(conv2,dim=3)
        conv3 = torch.squeeze(conv3,dim=3)
        
        x1 = self.max_pool1(conv1)
        x2 = self.max_pool2(conv2)
        x3 = self.max_pool3(conv3)
        
        x = torch.cat((x1,x2),dim=1)
        x = torch.cat((x,x3),dim=1).squeeze(dim=2)
        
        output = self.linear(self.dropout(x))
        
        return output 
 
def train(model,sentence,label):
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    steps = 0
    best_acc = 0
    model.train()
    print ("-"*80)
    print('Training....')
    
    for epoch in range(1,2): ##2个epoch
        for step,x in enumerate(torch.split(sentence,1,dim=0)):
            target = torch.zeros(1)
            target[0] = label[step]
            target = torch.tensor(target,dtype=torch.long)
            optimizer.zero_grad()
            output  = model(x)
            loss = criterion(output, target)
            #loss.backward()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if step % 2 == 0:
                result = torch.max(output,1)[1].view(target.size())
                corrects = (result.data == target.data).sum()
                accuracy = corrects*100.0/1 ####1 is batch size 
                print('Epoch:',epoch,'step:',step,'- loss: %.6f'% loss.data.item(),\
                      'acc: %.4f'%accuracy)
    return model
        
 
if __name__ == '__main__':
    test = sentence2id(sentence)
    test.sen2sen(sentence)
    word,words_num = test.countword()
    test.word2id()
    
    sen_train = test.sen2id(sentence)
    sen_test = test.sen2id(test_sentence)
   
   
    X_train = torch.LongTensor((padded(sen_train,0)))
    X_test = torch.LongTensor((padded(sen_test,0)))
 
    Embedding = ModelEmbeddings(words_num+1,10,0)
    
    X_train_embed = Embedding.Embedding(X_train)
    X_test_embed = Embedding.Embedding(X_test)
    print(X_train_embed.size())
    #print(X_test_embed.size())
    
    ## TextCNN
    textcnn = textCNN(words_num,10,2)
    model = train(textcnn,X_train_embed,Label)  
    print(torch.max(model(X_test_embed),1)[1])
