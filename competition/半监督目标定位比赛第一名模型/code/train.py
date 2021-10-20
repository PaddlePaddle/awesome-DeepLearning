import datetime
import paddle
import paddle.nn.functional as F
from paddle.io import DataLoader
import dataset
import numpy as np
import argparse
import os


# 参数设置 help 用简单的英文描述了改参数的作用，你需要修改的就是train_dataset，这里要换成你数据集的路径
def config():
    parser = argparse.ArgumentParser(description='train params')
    parser.add_argument('--Min_LR', default=0.0001, help='min lr')
    parser.add_argument('--Max_LR', default=0.01, help='max lr')
    parser.add_argument('--epoch', default=20, help='epoches')
    parser.add_argument('--mode_path', default=False, help='where your pretrained model')
    parser.add_argument('--train_bs', default=12, help='batch size for training')
    parser.add_argument('--test_bs', default=12, help='batch size for testing')
    parser.add_argument('--show_step', default=20, help='if step%show_step == 0 : print the info')
    parser.add_argument('--train_dataset', default=r'E:\Saliency\Dataset\DUST\DUTS-TR', help='where your train dataset')
    parser.add_argument('--save_path', default='weight', help='where you want to save the pdparams files')
    parser.add_argument('--save_iter', default=1, help=r'every iter to save model')
    cag = parser.parse_args()
    return cag


cag = config()


# 损失函数定义
def structure_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask)

    pred = F.sigmoid(pred)
    inter = (pred * mask).sum(axis=(2, 3))
    union = (pred + mask).sum(axis=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return bce + iou.mean()


# 训练过程
def train(Dataset, Network, savepath):
    # 设置数据
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    cfg = Dataset.Config(
        snapshot=cag.mode_path, datapath=cag.train_dataset, savepath=savepath,
        mode='train', batch=cag.train_bs, lr=cag.Max_LR, momen=0.9, decay=5e-4, epoch=cag.epoch
    )

    data = Dataset.Data(cfg)
    loader = DataLoader(
        data,
        batch_size=cfg.batch,
        shuffle=True,
        num_workers=12,
        use_shared_memory=False,
    )

    # 设置网络
    net = Network
    net.train()

    # 查看参数
    total_params = sum(p.numel() for p in net.parameters())
    print('total params : ', total_params)

    # 设置优化器和学习率衰减
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=cag.Max_LR, T_max=len(loader)*cag.epoch)
    optimizer = paddle.optimizer.Momentum(parameters=net.parameters(), learning_rate=scheduler, momentum=0.9,
                                          weight_decay=cfg.decay)
    global_step = 0

    # 开始训练
    all_losses = []
    all_lr = []
    for epoch in range(0, cfg.epoch):
        start = datetime.datetime.now()
        loss_list = []
        for batch_idx, (image, mask) in enumerate(loader, start=1):
            all_lr.append(optimizer.get_lr())
            optimizer.clear_grad()

            global_step += 1
            out = net(image)[0]
            loss = structure_loss(out, mask)

            loss_list.append(loss.numpy()[0])
            all_losses.append(loss.numpy()[0])
            loss.backward()
            optimizer.step()

            if batch_idx % cag.show_step == 0:
                msg = '%s | step:%d/%d/%d (%.2f%%) | lr=%.6f |  loss=%.6f | %s ' % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), batch_idx, epoch + 1, cfg.epoch,
                    batch_idx / (50000 / cag.train_bs) * 100, optimizer.get_lr(), loss.item()
                    , image.shape)
                print(msg)

        if epoch % cag.save_iter == 0:
            paddle.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1) + '.pdparams')

        end = datetime.datetime.now()
        spend = int((end - start).seconds)
        mins = spend // 60
        secon = spend % 60
        loss_list = '%.5f' % np.mean(loss_list)
        print(f'this epoch spend {mins} m {secon} s and the average loss is {loss_list}', '\n')


if __name__ == '__main__':
    from models import Res2NetandACFFNet
    from models import Res2NetandFMFNet
    from models import ResNeXtandACFFNet
    from models import SwinTandACFFNet

    model = Res2NetandACFFNet()
    train(dataset, model, 'weight')
