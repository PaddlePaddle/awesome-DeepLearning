import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from Net import LeNet_5

# paremeters
momentum = 0.9
record_step = 10
batch_size = 128
test_batch_size = 1000
lr = 0.001
Epoch = 1
random_seed = 1
torch.cuda.manual_seed_all(random_seed)

# test
def test(test_loader, net, loss_function):
    net.eval()
    total_loss = 0
    acc = 0
    for data, target in test_loader:
        data = torch.tensor(data).type(torch.FloatTensor).cuda()
        target = torch.tensor(target).type(torch.LongTensor).cuda()
        out = net(data)
        loss = loss_function(out, target)
        classification = torch.max(out,1)[1]
        total_loss += loss.item()
        correct = (target == classification).sum()
        acc+=correct.item()
    
    return total_loss/len(test_loader.dataset), acc/len(test_loader.dataset)   
    print(total_loss/len(test_loader.dataset))
    print(acc/len(test_loader.dataset))


# train
def main():
    # load dataset
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) 
                       ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True)
    print('Finished reading data')

    # load net
    net = LeNet_5().cuda()
    loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # train
    max_accuracy = 0
    accuracy_list = []
    for epoch in range(Epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            net.train()
            data = torch.tensor(data).type(torch.FloatTensor).cuda()
            target = torch.tensor(target).type(torch.LongTensor).cuda()
            out = net(data)
            classification = torch.max(out,1)[1]
            optimizer.zero_grad()
            loss = loss_function(out, target)
            loss.backward()
            optimizer.step()

            if (batch_idx+1)%(record_step) == 0:
                test_loss, accuracy = test(test_loader, net, loss_function)
                accuracy_list.append(accuracy)
                # save the best model
                if accuracy >= max_accuracy:
                    torch.save(net, './net.pkl')
                print(f'Epoch{epoch+1} iteration{batch_idx}/{len(train_loader)}: Accuracy = {accuracy} '
                      f'Loss_train = {loss.item()/len(data)} Loss_test = {test_loss}')

    print('Finished all!')
    # figure
    plt.plot(accuracy_list)
    plt.title('Accuracy Line')
    plt.xlabel('Time')
    plt.ylabel('Accuracy')
    plt.savefig("pic.png")
    plt.show()

if __name__ == '__main__':
    main()
