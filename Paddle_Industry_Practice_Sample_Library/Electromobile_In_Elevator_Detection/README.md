# �����ڵ�ƿ�����Ҽ��

## ����

* [��Ŀ˵��](#��Ŀ˵��)
* [��װ˵��](#��װ˵��)
* [����׼��](#����׼��)
* [ģ��ѡ��](#ģ��ѡ��)
* [ģ��ѵ��](#ģ��ѵ��)
* [ģ�͵���](#ģ�͵���)
* [����������](#����������)
* [�����ⲿ��](#�����ⲿ��)

<a name="��Ŀ˵��"></a>

����������ƿ����¥�뻧�����Ļ����¹��ż����ʣ���Ը������Ƴ�����Ӧ�ĵ�ƿ�����Ҽ��ģ�ͣ�ּ�ڴ�Դͷ������һ����ķ����� �������Ħ�г�ģ�Ϳ��ܻᷢ����������������˶����ͼ�������ʽʵ�ָ�Ϊ��ȷ��ʶ�� ������ʹ���˷ɽ�Ŀ�����׼�PaddleDetection�е�picodetģ���Լ�ͼ��ʶ���׼�PaddleClas�е�������ͨ��ʶ��ģ�͡�

![demo](docs/images/result_5.png)

ע:AI Studio�������д�����ο�[�����ڵ�ƿ�����ȫ����](https://aistudio.baidu.com/aistudio/projectdetail/3497217?channelType=0&channel=0)(����gpu��Դ)
## 2 ��װ˵��

##### ����Ҫ��

* PaddlePaddle = 2.2.2
* Python >= 3.5

<a name="����׼��"></a>

## 3 ����׼��

��������picodet��ģ�����ݼ�ΪVOC��ʽ(ʹ��labelimg�Ƴ�)������21903�ŵ����е�ͼƬ������ѵ����17522�ţ����Լ�4381�ţ��������ճ��ĵ��ݳ����У�����14715��Ħ�г��Ŀ�23058���˵Ŀ�3750�����г��Ŀ�����picodetʹ�õ���coco��ʽ��������Ҫ��VOC��ʽת����coco��ʽ�� ����VOC���ݼ���ʹ��python��labelimgͼ���ע����ΪԭʼͼƬ���ɶ�Ӧ�ı�עxml�ļ���Ϊԭʼ��VOC��ʽ���ݼ������ɵ�xml�ļ���ʽ����ͼ��ʾ������ÿ��object��������ÿһ������object�е�name������������ֶ�bndbox�а�����ľ������꣨���Ͻ��Լ����½ǣ���

![label_img][docs/images/label_img.png]

![xml_content](docs/images/xml_content.png)


����VOC���ݼ��� ���ͼƬ��ע����һ�������������ݼ�����ÿ��ͼƬ����xml��Ӧ�������������ɶ�Ӧ��ѵ�����Լ����Լ�.

```
������ classify_voc.py
������ picodet_motorcycle
��   ������ Annotations
��   ��   ������ 1595214506200933-1604535322-[]-motorcycle.xml
��   ��   ������ 1595214506200933-1604542813-[]-motorcycle.xml
��   ��   ������ 1595214506200933-1604559538-[]-motorcycle.xml
|   ...
��   ������ ImageSets
��   ��   ������ Main
��   ��       ������ test.txt
��   ��       ������ train.txt
��   ��       ������ trainval.txt
��   ��       ������ val.txt
��   ������ JPEGImages
��       ������ 1595214506200933-1604535322-[]-motorcycle.jpg
��       ������ 1595214506200933-1604542813-[]-motorcycle.jpg
��       ������ 1595214506200933-1604559538-[]-motorcycle.jpg
��       |   ...
������ picodet_motorcycle.zip
������ prepare_voc_data.py
������ test.txt
������ trainval.txt
```

VOC���ݼ� [���ص�ַ](https://aistudio.baidu.com/aistudio/datasetdetail/128282)
���������ݼ� [���ص�ַ](https://aistudio.baidu.com/aistudio/datasetdetail/128448)
��VOC��ʽ�����ݼ�ת��Ϊcoco��ʽ��ʹ��paddle�Դ���ת���ű�����
������˵����ʹ��ʱ���޸�·��
```
python x2coco.py --dataset_type voc  --voc_anno_dir /home/aistudio/data/data128282/ --voc_anno_list /home/aistudio/data/data128282/trainval.txt --voc_label_list /home/aistudio/data/data128282/label_list.txt  --voc_out_name voc_train.json
python x2coco.py --dataset_type voc --voc_anno_dir /home/aistudio/data/data128282/ --voc_anno_list /home/aistudio/data/data128282/test.txt --voc_label_list /home/aistudio/data/data128282/label_list.txt --voc_out_name voc_test.json
mv voc_test.json /home/aistudio/data/data128282/
mv voc_train.json /home/aistudio/data/data128282/

```
<a name="ģ��ѡ��"></a>

## 4 ģ��ѡ��

������ѡ����PaddleDetection�������ȫ�µ�������ϵ��ģ��PP-PicoDet

PP-PicoDetģ���������ص㣺

  - ���ߵ�mAP: ��һ����1M������֮��mAP(0.5:0.95)��Խ30+(����416����ʱ)��
  - �����Ԥ���ٶ�: ����Ԥ����ARM CPU�¿ɴ�150FPS��
  - �����Ѻ�: ֧��PaddleLite/MNN/NCNN/OpenVINO��Ԥ��⣬֧��ת��ONNX���ṩ��C++/Python/Android��demo��
  - �Ƚ����㷨: ������SOTA�㷨�н����˴���, ������ESNet, CSP-PAN, SimOTA�ȵȡ�


<a name="ģ��ѵ��"></a>

## 5 ģ��ѵ��


���Ȱ�װ������
```
cd code/train/
pip install pycocotools
pip install faiss-gpu
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple  
```

����Ϊservingģ�͵�׼��
```
pip install paddle-serving-app==0.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddle-serving-client==0.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddle-serving-server-gpu==0.6.3.post102 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<a name="ģ�͵���"></a>

## 6 ģ�͵���


����Ϊservingģ��
```
cd code/train/
python export_model.py  --export_serving_model=true -c picodet_lcnet_1_5x_416_coco.yml --output_dir=./output_inference/
```

```
cd code/train/output_inference/picodet_lcnet_1_5x_416_coco/
mv serving_server/ code/picodet_lcnet_1_5x_416_coco/
```

��������
```
cd /home/aistudio/work/code/picodet_lcnet_1_5x_416_coco/
python3 web_service.py
```

����������ͼ��ʾ:

![infer_result](docs/images/infer_result.png)

<a name="����������"></a>

## 7 ����������

��Ŀ����ģ�Ͳ�����Ϻ��ƿ�����Ҽ��Ĺ��ܱ��Ͷ��ʹ�ã���Ϊ����������׼ȷ�ȼ���������Ҫһ������ļ�����ʽ�����������PaddleClas��ͼ��ʶ���е�������ͨ��ʶ��ģ��general_PPLCNet_x2_5_lite_v1.0_infer

���ȴ�paddle���ؽ�ѹģ�Ͳ�����Ϊservingģ��
```
cd code/
wget -P models/ https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/rec/models/inference/general_PPLCNet_x2_5_lite_v1.0_infer.tar
cd models
tar -xf general_PPLCNet_x2_5_lite_v1.0_infer.tar
python3 -m paddle_serving_client.convert --dirname ./general_PPLCNet_x2_5_lite_v1.0_infer/ --model_filename inference.pdmodel  --params_filename inference.pdiparams --serving_server ./general_PPLCNet_x2_5_lite_v1.0_serving/  --serving_client ./general_PPLCNet_x2_5_lite_v1.0_client/
cp  -r ./general_PPLCNet_x2_5_lite_v1.0_serving ../general_PPLCNet_x2_5_lite_v1.0/
```

��ѹ���ݼ������·���޸�make_label.py,�������������.
```
cd code
python make_label.py
python python/build_gallery.py -c build_gallery/build_general.yaml -o IndexProcess.data_file="./index_label.txt" -o IndexProcess.index_dir="index_result"
mv index_result/ general_PPLCNet_x2_5_lite_v1.0/
```

<a name="�����ⲿ��"></a>

## 7 �����ⲿ��
```
cd /home/aistudio/work/code/general_PPLCNet_x2_5_lite_v1.0/
python recognition_web_service_onlyrec.py
```

��ʵ�ʳ�������������ͼ��ʾ��

![index_infer_result](docs/images/index_infer_result.png)
