SqueezeNet

模型介绍

《SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and < 0.5MB model size》
提出了名为SqueezeNet的轻量级CNN，达到AlexNet级别的精度，参数仅为AlexNet的1/50。采用Deep
Compression模型压缩技术，将SqueezeNet压缩为不到0.5MB的模型。
	SqueezeNet以一个标准卷积层开始，随后接8个Fire module，以一个标准卷积层结束。作者从
开始到结束逐渐增加滤波器的数量。SqueezeNet在conv1、fire4、fire8和conv10后面接stride为2
的max-pooling。

	SqueezeNet的压缩策略：
	
	1.将3* 3卷积替换成 1*1 卷积：通过这一步，一个卷积操作的参数数量减少了9倍
	2.减少3*3 卷积的通道数：一个 3*3 卷积的计算量是 3*3*M*N（其中 M， N 分别是输入Feature
	Map和输出Feature Map的通道数），作者任务这样一个计算量过于庞大，因此希望将 M，N 减小
	以减少参数数量；
	3.将降采样后置：作者认为较大的Feature Map含有更多的信息，因此将降采样往分类层移动。
	注意这样的操作虽然会提升网络的精度，但是它有一个非常严重的缺点：即会增加网络的计算量。
	
	Fire模块
	
	SqueezeNet是由若干个Fire模块结合卷积网络中卷积层，降采样层，全连接等层组成的。一个
	Fire模块由Squeeze部分和Expand部分组成（注意区分和Momenta的SENet[4]的区别）。Squeeze
	部分是一组连续的 1*1卷积组成，Expand部分则是由一组连续的 1*1卷积和一组连续的3*3 卷
	积cancatnate组成，因此 3*3 卷积需要使用same卷积。在Fire模块中，Squeeze部分1*1卷积的通
	道数记做s1x1，Expand部分 1*1卷积和3*3卷积的通道数分别记做 e1x1  和 e3x3（论文图画的不好，
	不要错误的理解成卷积的层数）。在Fire模块中，作者建议s1x1<e1x1+e3x3，这么做相当于在两个
	 3*3卷积的中间加入了瓶颈层，作者的实验中的一个策略是 s1x1=e1x1/4=e3x3/4 。

	实现：
	def fire_module(self, inputs, squeeze_depth, expand_depth, scope):
       	 with fluid.scope_guard(scope):
           	 squeeze =fluid.layers.conv2d(inputs, squeeze_depth, filter_size=1,
                                       stride=1, padding="VALID",
                                       act='relu', name="squeeze")
            	#print('squeeze shape:',squeeze.shape)
            	# squeeze
            	expand_1x1 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=1,
                                          stride=1, padding="VALID",
                                          act='relu', name="expand_1x1")
           	 #print('expand_1x1 shape:',expand_1x1.shape)

            	expand_3x3 = fluid.layers.conv2d(squeeze, expand_depth, filter_size=3,
                                          stride=1, padding=1,
                                          act='relu', name="expand_3x3")
            	#print('expand_3x3 shape:',expand_3x3.shape)
            	return fluid.layers.concat([expand_1x1, expand_3x3], axis=1)


	SqueezeNet的网络架构
	
	分为不加short-cut和加short-cut及short-cut跨有不同Feature Map个数的卷积的
	而且：
	激活函数默认都使用ReLU；
	fire9之后接了一个rate为0.5的dropout；
	使用same卷积。
	
	实现：代码较长这里就不贴出来了

实现步骤
	
	数据加载与预处理

	解压缩数据集，数据（图片)读取,以及定义数据读取器，注意验证集的标签是在csv
	文件中，所以这里和训练集的读取稍有不同，以及测试读取器是否顺利运行
	
	squeezenet构建
	
	先按照论文介绍，构建fire模块，接着构建整个模块

	训练模型

	获取测试集，然后对所构建的模型进行训练
	 out = net(imgs)
                loss = loss_fn(out, labels)
                acc = paddle.metric.accuracy(out, labels)
	 if batch_id % 5 == 0:
                    test_acc = get_validation_acc(net, valid_loader)
                    # 加入visualDL可视化
                    iteration += 1
                    writer.add_scalar(tag='acc', step=iteration, value
	      =acc)
                    writer.add_scalar(tag='test_acc', step=iteration, 
	      value=test_acc)
	 if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    paddle.jit.save(
                        layer = net,
                        path = modle_path,
                        input_spec = [InputSpec(shape=[None, 3, 
	         224, 224], dtype='float32')]
	
	验证模型·
	
	加载模型并计算测试集准确率
               
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

	SqueezeNet的压缩策略是依靠将 3*3卷积替换成 1*1
	卷积来达到的，其参数数量是等性能的AlexNet的2.14%
	。从参数数量上来看，SqueezeNet的目的达到了。
	SqueezeNet的最大贡献在于其开拓了模型压缩这一方向
	，之后的一系列文章也就此打开。
	这里我们着重说一下SqueezeNet的缺点：
	1.SqueezeNet的侧重的应用方向是嵌入式环境，目前嵌入
	式环境主要问题是实时性。SqueezeNet的通过更深的深
	度置换更少的参数数量虽然能减少网络的参数，但是其丧
	失了网络的并行能力，测试时间反而会更长，这与目前的
	主要挑战是背道而驰的；
	2.虽然纸面上是减少了50倍的参数，但是问题的主要症结
	在于AlexNet本身全连接节点过于庞大，50倍参数的减少
	和SqueezeNet的设计并没有关系，考虑去掉全连接之后
	3倍参数的减少更为合适
 	最后通过该实训了解squeezenet的结构，熟练了该模型
	的使用。
飞桨项目地址：
https://aistudio.baidu.com/aistudio/projectdetail/2259347

	
	

	
	

	
	
	


	