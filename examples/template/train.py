# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding:utf-8
import os
import sys
import paddle.v2 as paddle
from vgg import vgg_bn_drop
from resnet import resnet_cifar10
 
class TestCIFAR:
    # ***********************��ʼ������***************************************
    def __init__(self):
        # ��ʼ��paddpaddle,ֻ����CPU,��GPU�ر�
        paddle.init(use_gpu=False, trainer_count=2)
 
    # **********************��ȡ����***************************************
    def get_parameters(self, parameters_path=None, cost=None):
        if not parameters_path:
            # ʹ��cost����parameters
            if not cost:
                print "������cost����"
            else:
                # ������ʧ������������
                parameters = paddle.parameters.create(cost)
                return parameters
        else:
            # ʹ��֮ǰѵ���õĲ���
            try:
                # ʹ��ѵ���õĲ���
                with open(parameters_path, 'r') as f:
                    parameters = paddle.parameters.Parameters.from_tar(f)
                return parameters
            except Exception as e:
                raise NameError("��Ĳ����ļ�����,����������:%s" % e)
 
    # ***********************��ȡѵ����***************************************
    def get_trainer(self):
        # ���ݴ�С
        datadim = 3 * 32 * 32
 
        # ���ͼƬ���ڵ���Ϣ��ǩ
        lbl = paddle.layer.data(name="label",
                                type=paddle.data_type.integer_value(10))
 
        # ��ȡȫ���Ӳ�,Ҳ���Ƿ�����
        #
        out = vgg_bn_drop(datadim=datadim)
        # out = resnet_cifar10(datadim=datadim)
 
        # �����ʧ����
        cost = paddle.layer.classification_cost(input=out, label=lbl)
 
        # ʹ��֮ǰ����õĲ����ļ���ò���
        # parameters = self.get_parameters(parameters_path="../model/model.tar")
        # ʹ����ʧ�������ɲ���
        parameters = self.get_parameters(cost=cost)
 
        '''        �����Ż�����
        learning_rate �������ٶ�
        momentum ��ǰ�涯���Ż��ı���
        regularzation ����,��ֹ�����
        '''
        momentum_optimizer = paddle.optimizer.Momentum(
            momentum=0.9,
            regularization=paddle.optimizer.L2Regularization(rate=0.0002 * 128),
            learning_rate=0.1 / 128.0,
            learning_rate_decay_a=0.1,
            learning_rate_decay_b=50000 * 100,
            learning_rate_schedule="discexp")
 
        '''
        ����ѵ����
        cost ������
        parameters ѵ������,����ͨ������,Ҳ����ʹ��֮ǰѵ���õĲ���
        update_equation �Ż�����
        '''
        trainer = paddle.trainer.SGD(cost=cost,
                                     parameters=parameters,
                                     update_equation=momentum_optimizer)
        return trainer
 
    # ***********************��ʼѵ��***************************************
    def start_trainer(self):
        # �������
        reader = paddle.batch(reader=paddle.reader.shuffle(reader=paddle.dataset.cifar.train10(),
                                                           buf_size=50000),
                              batch_size=128)
 
        # ָ��ÿ�����ݺ�padd.layer.data�Ķ�Ӧ��ϵ
        feeding = {"image": 0, "label": 1}
 
        # ����ѵ���¼��������־
        def event_handler(event):
            if isinstance(event, paddle.event.EndIteration):
                if event.batch_id % 100 == 0:
                    print "\nPass %d, Batch %d, Cost %f, %s" % (
                        event.pass_id, event.batch_id, event.cost, event.metrics)
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
 
            # ÿһ��ѵ�����֮��
            if isinstance(event, paddle.event.EndPass):
                # ����ѵ���õĲ���
                model_path = 'model'
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                with open(model_path + '/modelCifar.tar', 'w') as f:
                    trainer.save_parameter_to_tar(f)
 
                # ����׼ȷ��
                result = trainer.test(reader=paddle.batch(reader=paddle.dataset.cifar.test10(),
                                                          batch_size=128),
                                      feeding=feeding)
                print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
 
        # ��ȡѵ����
        trainer = self.get_trainer()
 
        '''
        ��ʼѵ��
        reader ѵ������
        num_passes ѵ��������
        event_handler ѵ�����¼�,������ѵ����ʱ��Ҫ��һЩʲô����
        feeding ˵��ÿ�����ݺ�padd.layer.data�Ķ�Ӧ��ϵ
        '''
        trainer.train(reader=reader,
                      num_passes=1,
                      event_handler=event_handler,
                      feeding=feeding)
 
if __name__ == '__main__':
    testCIFAR = TestCIFAR()
    # ��ʼѵ��
    testCIFAR.start_trainer()