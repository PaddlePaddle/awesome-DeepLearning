DenseNet

论文：Densely Connected Convolutional Networks
论文链接：https://arxiv.org/pdf/1608.06993.pdf
代码的github链接：https://github.com/liuzhuang13/DenseNet
MXNet版本代码（有ImageNet预训练模型）: https://github.com/miraclewkf/DenseNet

该文章是CVPR2017的oral。文章提出的DenseNet（Dense Convolutional Network）主要还是和ResNet及Inception网络做对比，思想上有借鉴，但却是全新的结构，网络结构并不复杂，却非常有效！最近一两年卷积神经网络提高效果的方向，要么深（比如ResNet，解决了网络深时候的梯度消失问题）要么宽（比如GoogleNet的Inception），而作者则是从feature入手，通过对feature的极致利用达到更好的效果和更少的参数。

DenseNet的几个优点：
1、减轻了vanishing-gradient（梯度消失）
2、加强了feature的传递
3、更有效地利用了feature
4、一定程度上较少了参数数量

	SqueezeNet的网络特点：
	
	深度学习网络中，随着网络深度的加深，梯度消失的问题会越来越明显。ResNet，Highway Networks，Stochastic depth，FractalNets等网络都在不同方面针对这个问题提出解决方案，但核心方法都是建立浅层与深层之间的连接。
	DenseNet继续延申了这一思想，将当前层与之前所有层连接起来。 DenseNet的一个优点就是网络更窄，参数更少，并且特征和梯度的传递更有效，网络也就更容易训练。
	
	网络整体结构：

	采用DenseNet-121结构
	包含3个Dense Block,每个Dense Block中的所有层都与其之前的每一层相连。每两个Dense Block之间还有一个1×11 \times 11×1卷积层和一个2×22 \times 22×2池化层,这是为了减少输入的feature map，降维减少计算量，融合各通道特征。 每张图片先经过卷积输入，然后经过几个Dense Block，最后再经过一次卷积输入到全连接层中，输出（分类）结果。

	实现：
class DenseNet(): 
    def __init__(self, layers, dropout_prob):
        self.layers = layers
        self.dropout_prob = dropout_prob
 
    def bottleneck_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num * 4, filter_size=1, name=name + '_conv1')
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)

        bn = fluid.layers.batch_norm(input=dropout, act='relu', name=name + '_bn2')
        conv2 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=3, padding=1, name=name + '_conv2')
        dropout = fluid.layers.dropout(x=conv2, dropout_prob=self.dropout_prob)

        return dropout

    def dense_block(self, input, block_num, fliter_num, name):
        layers = []
        layers.append(input)#拼接到列表

        x = self.bottleneck_layer(input, fliter_num, name=name + '_bottle_' + str(0))
        layers.append(x)
        for i in range(block_num - 1):
            x = paddle.fluid.layers.concat(layers, axis=1)
            x = self.bottleneck_layer(x, fliter_num, name=name + '_bottle_' + str(i + 1))
            layers.append(x)

        return paddle.fluid.layers.concat(layers, axis=1)

    def transition_layer(self, input, fliter_num, name):
        bn = fluid.layers.batch_norm(input=input, act='relu', name=name + '_bn1')
        conv1 = fluid.layers.conv2d(input=bn, num_filters=fliter_num, filter_size=1, name=name + '_conv1') 
        dropout = fluid.layers.dropout(x=conv1, dropout_prob=self.dropout_prob)
        
        return fluid.layers.pool2d(input=dropout, pool_size=2, pool_type='avg', pool_stride=2)
 
    def net(self, input, class_dim=2): 

        layer_count_dict = {
            121: (32, [6, 12, 24, 16]),
            169: (32, [6, 12, 32, 32]),
            201: (32, [6, 12, 48, 32]),
            161: (48, [6, 12, 36, 24])
        }
        layer_conf = layer_count_dict[self.layers]

        conv = fluid.layers.conv2d(input=input, num_filters=layer_conf[0] * 2, 
            filter_size=7, stride=2, padding=3, name='densenet_conv0')
        conv = fluid.layers.pool2d(input=conv, pool_size=3, pool_padding=1, pool_type='max', pool_stride=2)
        for i in range(len(layer_conf[1]) - 1):
            conv = self.dense_block(conv, layer_conf[1][i], layer_conf[0], 'dense_' + str(i))
            conv = self.transition_layer(conv, layer_conf[0], name='trans_' + str(i))

        conv = self.dense_block(conv, layer_conf[1][-1], layer_conf[0], 'dense_' + str(len(layer_conf[1])))
        conv = fluid.layers.pool2d(input=conv, global_pooling=True, pool_type='avg')
        out = fluid.layers.fc(conv, class_dim, act='softmax')
        # last fc layer is "out" 
        return out

实现步骤
	
	数据加载与预处理
	iChallenge-PM是百度大脑和中山大学中山眼科中心联合举办的iChallenge比赛中，提供的关于病理性近视（Pathologic Myopia，PM）的医疗类数据集，包含1200个受试者的眼底视网膜图片，训练、验证和测试数据集各400张。
	training.zip：包含训练中的图片和标签。
	validation.zip：包含验证集的图片。
	valid_gt.zip：包含验证集的标签。
	iChallenge-PM中既有病理性近视患者的眼底图片，也有非病理性近视患者的图片，命名规则如下：
	病理性近视（PM）：文件名以P开头。
	非病理性近视（non-PM）： 高度近似（high myopia）：文件名以H开头。
	正常眼睛（normal）：文件名以N开头。
	处理数据集：首先解压数据集，根据数据集介绍，训练集images通过文件名获取相应的labels,测试集通过读取PM_Label_and_Fovea_Location.xlsx文件获取文件名和label信息，然后分别生成traindata.txt和valdata.txt文件。使用时直接读入文件即可获取图片与标签对应关系。
	
	DenseNet构建
	
	主要分了两部分dense_block与transition_layer，一个实现了denseNet的每一个模块的功能，另一个完成每一个模块的相连（BN+Relu+卷积+池化）
	再完成模块之间的连接

	自定义用户图片读取器，先初始化图片种类，数量，定义图片增强和强制缩放函数
def custom_image_reader(file_list, data_dir, mode): 
    """ 
    自定义用户图片读取器，先初始化图片种类，数量 
    :param file_list: 
    :param data_dir: 
    :param mode: 
    :return: 
    """ 
    with codecs.open(file_list) as flist: 
        lines = [line.strip() for line in flist] 
 
    def reader(): 
        np.random.shuffle(lines) 
        for line in lines: 
            if mode == 'train' or mode == 'val': 
                img_path, label = line.split() 
                img = Image.open(img_path)
                try: 
                    if img.mode != 'RGB': 
                        img = img.convert('RGB') 
                    if train_parameters['image_enhance_strategy']['need_distort'] == True: 
                        img = distort_color(img) 
                    if train_parameters['image_enhance_strategy']['need_rotate'] == True: 
                        img = rotate_image(img) 
                    if train_parameters['image_enhance_strategy']['need_crop'] == True: 
                        img = random_crop(img, train_parameters['input_size']) 
                    if train_parameters['image_enhance_strategy']['need_flip'] == True: 
                        mirror = int(np.random.uniform(0, 2)) 
                        if mirror == 1: 
                            img = img.transpose(Image.FLIP_LEFT_RIGHT) 
                    # HWC--->CHW && normalized 
                    img = np.array(img).astype('float32') 
                    img -= train_parameters['mean_rgb'] 
                    img = img.transpose((2, 0, 1))  # HWC to CHW 
                    img *= 0.007843                 # 像素值归一化 
                    yield img, int(label) 
                except Exception as e: 
                    pass                            # 以防某些图片读取处理出错，加异常处理 
            elif mode == 'test': 
                img_path = os.path.join(data_dir, line) 
                img = Image.open(img_path) 
                if img.mode != 'RGB': 
                    img = img.convert('RGB') 
                img = resize_img(img, train_parameters['input_size']) 
                # HWC--->CHW && normalized 
                img = np.array(img).astype('float32') 
                img -= train_parameters['mean_rgb'] 
                img = img.transpose((2, 0, 1))  # HWC to CHW 
                img *= 0.007843  # 像素值归一化 
                yield img 
 
    return reader 

	训练模型

	获取测试集，然后对所构建的模型进行训练
	
	评估模型·
def eval_all():     
    eval_file_path = os.path.join(data_dir, eval_file)     
    total_count = 0     
    right_count = 0     
    with codecs.open(eval_file_path, encoding='utf-8') as flist:      
        lines = [line.strip() for line in flist]     
        t1 = time.time()     
        for line in lines:     
            total_count += 1     
            parts = line.strip().split()     
            result = infer(parts[0])     
            print(parts[0]+"infer result:{0} answer:{1}".format(result, parts[1]))     
            if str(result) == parts[1]:     
                right_count += 1     
        period = time.time() - t1     
        print("total eval count:{0} cost time:{1} predict accuracy:{2}".format(total_count, "%2.2f sec" % period, right_count / total_count))
	
	直接使用如下方式进行评估
	if __name__ == '__main__':     
    		eval_all()

	预测函数：
	def infer_img(imgpath)
	直接使用如下方式就可以预测图片了（imgpath为图片路径）
	imgpath = 'PALM-Training400/P0001.jpg'
	infer_img(imgpath)
               
	def evaluation(net, valid_loader):
    	accs = []
    	for batch_id, (imgs, labels) in enumerate(valid_loader()):
       	# print(imgs.shape)
      	 out = net(imgs)
        	acc = paddle.metric.accuracy(out, labels).numpy()[0]
        	accs.append(acc)
    	acc = np.array(accs).mean()
    	return acc    

总结

	DenseNet和ResNet的核心思想都是创建一个跨层连接来连通网络的前后层，在DenseNet中作者为了最大化层级之间的信息流，将所有层两两进行连接，这也是DenseNet(Densely Connected Convolutional Networks)名字的意义所在，密集的网络连接。

飞桨项目地址：
https://aistudio.baidu.com/aistudio/projectdetail/2258065

	
	

	
	

	
	
	


	