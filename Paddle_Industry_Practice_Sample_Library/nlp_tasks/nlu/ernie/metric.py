# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


from collections import Counter
import numpy as np
import paddle

class SeqEntityScore(object):
    def __init__(self, id2tag):
        self.id2tag = id2tag
        self.real_entities = []
        self.pred_entities = []
        self.correct_entities = []
        
    def reset(self):
        self.real_entities.clear()
        self.pred_entities.clear()
        self.correct_entities.clear()

    def compute(self, real_count, pred_count, correct_count):
        recall = 0 if real_count == 0 else (correct_count / real_count)
        precision = 0 if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def get_result(self):
        result = {}
        real_counter = Counter([item[0] for item in self.real_entities])
        pred_counter = Counter([item[0] for item in self.pred_entities])
        correct_counter = Counter([item[0] for item in self.correct_entities])
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter.get(label, 0)
            correct_count = correct_counter.get(label, 0)
            recall, precision, f1 = self.compute(real_count, pred_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        real_total_count = len(self.real_entities)
        pred_total_count = len(self.pred_entities)
        correct_total_count = len(self.correct_entities)
        recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result
    
    def get_entities_bios(self, seq):
        entities = []
        entity = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                if isinstance(tag, paddle.Tensor):
                    tag = tag.numpy()[0]
                tag = self.id2tag[tag]

            if tag.startswith("S-"):
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
                entity[1] = indx
                entity[2] = indx
                entity[0] = tag.split('-')[1]
                entities.append(entity)
                entity = (-1, -1, -1)
            if tag.startswith("B-"):
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
                entity[1] = indx
                entity[0] = tag.split('-')[1]
            elif tag.startswith('I-') and entity[1] != -1:
                _type = tag.split('-')[1]
                if _type == entity[0]:
                    entity[2] = indx
                if indx == len(seq) - 1:
                    entities.append(entity)
            else:
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
        return entities

    def get_entities_bio(self, seq):
        entities = []
        entity = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = self.id2tag[tag]
                
            if tag.startswith("B+"):
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
                entity[1] = indx
                entity[0] = tag.split('+', maxsplit=1)[1]
                entity[2] = indx
                if indx == len(seq) - 1:
                    entities.append(entity)
            elif tag.startswith('I+') and entity[1] != -1:
                _type = tag.split('+', maxsplit=1)[1]
                if _type == entity[0]:
                    entity[2] = indx
                if indx == len(seq) - 1:
                    entities.append(entity)
            else:
                if entity[2] != -1:
                    entities.append(entity)
                entity = [-1, -1, -1]
        return entities

    def update(self, real_paths, pred_paths):
        
        if isinstance(real_paths, paddle.Tensor):
            real_paths = real_paths.numpy()
        if isinstance(pred_paths, paddle.Tensor):
            pred_paths = pred_paths.numpy()

        for real_path, pred_path in zip(real_paths, pred_paths):
            real_ents = self.get_entities_bio(real_path)
            pred_ents = self.get_entities_bio(pred_path)
            self.real_entities.extend(real_ents)
            self.pred_entities.extend(pred_ents)
            self.correct_entities.extend([pred_ent for pred_ent in pred_ents if pred_ent in real_ents])

    def format_print(self, result, print_detail=False):
        def print_item(entity, metric):
            if entity != "Total":
                print(f"Entity: {entity} - Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")
            else:
                print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")

        print_item("Total", result["Total"])
        if print_detail:
            for key in result.keys():
                if key == "Total":
                    continue
                print_item(key, result[key])
            print("\n")


class MultiLabelClassificationScore(object):
    def __init__(self, id2label):
        self.id2label = id2label
        self.all_pred_labels = []
        self.all_real_labels = []
        self.all_correct_labels = []        
    
    def reset(self):
        self.all_pred_labels.clear()
        self.all_real_labels.clear()
        self.all_correct_labels.clear()
     
    def update(self, pred_labels, real_labels):
        if not isinstance(pred_labels, list):
            pred_labels = pred_labels.numpy().tolist()
        if not isinstance(real_labels, list):
            real_labels = real_labels.numpy().tolist()

        for i in range(len(real_labels)):
            for j in range(len(real_labels[0])):
                if real_labels[i][j] == 1 and  pred_labels[i][j] > 0:
                    self.all_correct_labels.append(self.id2label[j])
                if real_labels[i][j] == 1:
                    self.all_real_labels.append(self.id2label[j])
                if pred_labels[i][j] > 0:
                    self.all_pred_labels.append(self.id2label[j])

    def compute(self, pred_count , real_count, correct_count):
        recall  = 0. if real_count == 0 else (correct_count / real_count)
        precision = 0. if pred_count == 0 else (correct_count / pred_count)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def get_result(self):
        result = {}
        pred_counter = Counter(self.all_pred_labels)
        real_counter = Counter(self.all_real_labels)
        correct_counter = Counter(self.all_correct_labels)
        for label, count in real_counter.items():
            real_count = count
            pred_count = pred_counter[label]
            correct_count = correct_counter[label]
            precision, recall, f1 = self.compute(pred_count, real_count, correct_count)
            result[label] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}
        real_total_count = len(self.all_real_labels)
        pred_total_count = len(self.all_pred_labels)
        correct_total_count = len(self.all_correct_labels)
        recall, precision, f1 = self.compute(real_total_count, pred_total_count, correct_total_count)
        result["Total"] = {"Precision": round(precision, 4), 'Recall': round(recall, 4), 'F1': round(f1, 4)}

        return result         

    def format_print(self, result, print_detail=False):
        def print_item(entity, metric):
            if entity != "Total":
                print(f"Entity: {entity} - Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")
            else:
                print(f"Total: Precision: {metric['Precision']} - Recall: {metric['Recall']} - F1: {metric['F1']}")

        print_item("Total", result["Total"])
        if print_detail:
            for key in result.keys():
                if key == "Total":
                    continue
                print_item(key, result[key])
            print("\n") 
  

