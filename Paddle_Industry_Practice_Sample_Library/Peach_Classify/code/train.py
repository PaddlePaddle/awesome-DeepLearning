import paddle
import paddlehub.vision.transforms as T
import paddlehub as hub
from paddlehub.finetune.trainer import Trainer

from dataset import DemoDataset


def train():
    transforms = T.Compose(
            [T.Resize((256, 256)),
             T.CenterCrop(224),
             T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])],
            to_rgb=True)

    peach_train = DemoDataset(transforms)
    peach_validate =  DemoDataset(transforms, mode='val')

    model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["R0", "B1", "M2", "S3"])

    optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    trainer = Trainer(model, optimizer, checkpoint_dir='img_classification_ckpt', use_gpu=True) 
    trainer.train(peach_train, epochs=10, batch_size=16, eval_dataset=peach_validate, save_interval=1)


if __name__ == '__main__':
    train()
