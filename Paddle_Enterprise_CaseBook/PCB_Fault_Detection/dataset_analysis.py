import json
from collections import defaultdict
import matplotlib.pyplot as plt

with open("/home/aistudio/work/PCB_DATASET/Annotations/train.json") as f:
    data = json.load(f)

imgs = {}
for img in data['images']:
    imgs[img['id']] = {
        'h': img['height'],
        'w': img['width'],
        'area': img['height'] * img['width'],
    }

hw_ratios = []
area_ratios = []
label_count = defaultdict(int)
for anno in data['annotations']:
    hw_ratios.append(anno['bbox'][3]/anno['bbox'][2])
    area_ratios.append(anno['area']/imgs[anno['image_id']]['area'])
    label_count[anno['category_id']] += 1

print(label_count, len(data['annotations']) / len(data['images']))

plt.hist(hw_ratios, bins=100, range=[0, 2])
plt.show()

plt.hist(area_ratios, bins=100, range=[0, 0.005])
plt.show()