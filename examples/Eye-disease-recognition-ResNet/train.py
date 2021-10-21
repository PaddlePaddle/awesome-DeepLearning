import os
import paddle

from model.ResNet import ResNet
from model.save import save
from trianer.train_pm import train_pm


# 超参数的设置
use_gpu=False
lr=0.0001
momentum=0.9
load_model=True
save_model=True
EPOCH_NUM=20

# 版本参数的设置
model_version='O'

filedir=os.getcwd() #获取文件当前的主路径
model=ResNet(layers=50,class_dim=2,version=model_version)
if os.path.exists(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams') and load_model:
    model_params=paddle.load(f'./model/resnet{model.layers}_v{model_version}_PALM.pdparams')
    model.set_state_dict(model_params) # 加载预训练模型参数
annotion_path=filedir+'/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx' # 获取验证集标签数据地址
optimizer=paddle.optimizer.Momentum(learning_rate=lr,momentum=momentum,parameters=model.parameters())# 选择优化器

print('文件主路径：',filedir)
print('训练模型版本：',model_version)
print('是否采用预训练模型：',load_model)
print('是否采用GPU：',use_gpu)

if save_model: # 判断是否需要保存模型参数
    save=save
else:
    save=None
train_pm(model,filedir,annotion_path,optimizer,EPOCH_NUM=EPOCH_NUM,use_gpu=use_gpu,save=save)

# 版本参数的设置
model_version='B'

filedir=os.getcwd() #获取文件当前的主路径
model=ResNet(layers=50,class_dim=2,version=model_version)
if os.path.exists(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams') and load_model:
    model_params=paddle.load(f'./model/resnet{model.layers}_v{model_version}_PALM.pdparams')
    model.set_state_dict(model_params) # 加载预训练模型参数
annotion_path=filedir+'/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx' # 获取验证集标签数据地址
optimizer=paddle.optimizer.Momentum(learning_rate=lr,momentum=momentum,parameters=model.parameters())# 选择优化器

print('文件主路径：',filedir)
print('训练模型版本：',model_version)
print('是否采用预训练模型：',load_model)
print('是否采用GPU：',use_gpu)

if save_model: # 判断是否需要保存模型参数
    save=save
else:
    save=None
train_pm(model,filedir,annotion_path,optimizer,EPOCH_NUM=EPOCH_NUM,use_gpu=use_gpu,save=save)

# 版本参数的设置
model_version='C'

filedir=os.getcwd() #获取文件当前的主路径
model=ResNet(layers=50,class_dim=2,version=model_version)
if os.path.exists(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams') and load_model:
    model_params=paddle.load(f'./model/resnet{model.layers}_v{model_version}_PALM.pdparams')
    model.set_state_dict(model_params) # 加载预训练模型参数
annotion_path=filedir+'/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx' # 获取验证集标签数据地址
optimizer=paddle.optimizer.Momentum(learning_rate=lr,momentum=momentum,parameters=model.parameters())# 选择优化器

print('文件主路径：',filedir)
print('训练模型版本：',model_version)
print('是否采用预训练模型：',load_model)
print('是否采用GPU：',use_gpu)

if save_model: # 判断是否需要保存模型参数
    save=save
else:
    save=None
train_pm(model,filedir,annotion_path,optimizer,EPOCH_NUM=EPOCH_NUM,use_gpu=use_gpu,save=save)

# 版本参数的设置
model_version='D'

filedir=os.getcwd() #获取文件当前的主路径
model=ResNet(layers=50,class_dim=2,version=model_version)
if os.path.exists(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams') and load_model:
    model_params=paddle.load(f'./model/resnet{model.layers}_v{model_version}_PALM.pdparams')
    model.set_state_dict(model_params) # 加载预训练模型参数
annotion_path=filedir+'/PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx' # 获取验证集标签数据地址
optimizer=paddle.optimizer.Momentum(learning_rate=lr,momentum=momentum,parameters=model.parameters())# 选择优化器

print('文件主路径：',filedir)
print('训练模型版本：',model_version)
print('是否采用预训练模型：',load_model)
print('是否采用GPU：',use_gpu)

if save_model: # 判断是否需要保存模型参数
    save=save
else:
    save=None
train_pm(model,filedir,annotion_path,optimizer,EPOCH_NUM=EPOCH_NUM,use_gpu=use_gpu,save=save)
