import paddle

from model.ResNet import ResNet
from valider.valid_pm import valid_pm
from data_utils.valid_loader import valid_data_loader

# 标注路径配置和加载验证数据生成器
annotion_path='./PALM-Validation-GT/PM_Label_and_Fovea_Location.xlsx'
valid_loader=valid_data_loader('./PALM-Validation400',annotion_path)

# =================================================================
# 模型版本选择
model_version='O'
model_layers=50

# 模型的载入、模型参数的载入和配置
model=ResNet(layers=model_layers,class_dim=2,version=model_version)
model_params=paddle.load(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams')
model.set_state_dict(model_params)

# 模型的验证过程
valid_accuracy,valid_loss=valid_pm(model,valid_loader,batch_size=50)
print('[validation]:===model:ResNet{}-{}===accuracy:{:.5f}/loss:{:.5f}'.format(model.layers,model.version,valid_accuracy,valid_loss))

# =================================================================
# 模型版本选择
model_version='B'
model_layers=50

# 模型的载入、模型参数的载入和配置
model=ResNet(layers=model_layers,class_dim=2,version=model_version)
model_params=paddle.load(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams')
model.set_state_dict(model_params)

# 模型的验证过程
valid_accuracy,valid_loss=valid_pm(model,valid_loader,batch_size=50)
print('[validation]:===model:ResNet{}-{}===accuracy:{:.5f}/loss:{:.5f}'.format(model.layers,model.version,valid_accuracy,valid_loss))

# =================================================================
# 模型版本选择
model_version='C'
model_layers=50

# 模型的载入、模型参数的载入和配置
model=ResNet(layers=model_layers,class_dim=2,version=model_version)
model_params=paddle.load(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams')
model.set_state_dict(model_params)

# 模型的验证过程
valid_accuracy,valid_loss=valid_pm(model,valid_loader,batch_size=50)
print('[validation]:===model:ResNet{}-{}===accuracy:{:.5f}/loss:{:.5f}'.format(model.layers,model.version,valid_accuracy,valid_loss))

# =================================================================
# 模型版本选择
model_version='D'
model_layers=50

# 模型的载入、模型参数的载入和配置
model=ResNet(layers=model_layers,class_dim=2,version=model_version)
model_params=paddle.load(f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams')
model.set_state_dict(model_params)

# 模型的验证过程
valid_accuracy,valid_loss=valid_pm(model,valid_loader,batch_size=50)
print('[validation]:===model:ResNet{}-{}===accuracy:{:.5f}/loss:{:.5f}'.format(model.layers,model.version,valid_accuracy,valid_loss))