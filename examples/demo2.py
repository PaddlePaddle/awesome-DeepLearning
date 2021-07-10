import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.transforms as transforms

lr=0.01
momentum=0.5
log_interval=10
epochs=10
batch_size=64
test_batch_size=1000


train_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root='./data',train=True,download=False,
                  transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size = batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(
    datasets.MNIST(root='./data',train=False,
                   transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,),(0.3081,))])),
    batch_size = 64,shuffle=True)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,6,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc1 = nn.Linear(5*5*16,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return x


model = LeNet()


def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data=data.clone().detach().type(torch.FloatTensor)#.cuda()
        target=target.clone().detach().type(torch.LongTensor)#.cuda()
        torch.optim.SGD(model.parameters(),lr=lr,
                        momentum=momentum).zero_grad()
        output=model(data)
        loss=nn.CrossEntropyLoss()(output,target)
        loss.backward()
        torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum).step()
        if batch_idx%log_interval==0:
            print('Train Epoch:{}[{:5}/{}({:2.0f}%)]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()))



def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:
        data=data.clone().detach().type(torch.FloatTensor)#.cuda()
        target=target.clone().detach().type(torch.LongTensor)#.cuda()
        output=model(data)
        
        loss=nn.CrossEntropyLoss()(output,target)#把所有loss值进行累加
        test_loss+=loss
        
        classification = torch.max(output, 1)[1] #获得最大概率的下标
        
        class_y = classification.data.numpy() #对预测正确的数据个数进行累加
        correct += sum(class_y == target.data.numpy())
        
        test_loss=test_loss/len(data)
        #所有loss值进行累加，除以总的数据长度得平均loss
        
        print('Test set:Average loss:{:.4f},Accuracy:{:4}/{}({:2.0f}%)'.format(
        test_loss,correct,
            len(test_loader.dataset),100.*correct/len(test_loader.dataset)))


train(epochs)
print("\n")
test()
