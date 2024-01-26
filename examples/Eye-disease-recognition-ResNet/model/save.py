import paddle

# 构建模型保存函数
def save(accuracy,model):
    print('model save success !')
    if model==None:
        return
    model.max_accuracy=accuracy # 覆盖当前的最大正确率
    paddle.save(model.state_dict(),f'./model/resnet{model.layers}_v{model.version}_PALM.pdparams') # 保存模型
