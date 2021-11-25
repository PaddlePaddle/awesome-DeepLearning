import paddle
import paddlehub as hub
from PIL import Image
import matplotlib.pyplot as plt


def predict():
    model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["R0", "B1", "M2", "S3"])
    img_path = './dataset/test.jpg'
    img = Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    result = model.predict([img_path])
    print("桃子的类别被预测为:{}".format(result))


if __name__ == '__main__':
    predict()
