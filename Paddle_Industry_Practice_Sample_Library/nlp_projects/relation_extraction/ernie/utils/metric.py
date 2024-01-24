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


import paddle


class SPOMetric(paddle.metric.Metric):

    def __init__(self):
        super(SPOMetric, self).__init__()
        self.correct_count = 0.
        self.predict_count = 0.
        self.recall_count = 0.

    def update(self, batch_examples, batch_pred_examples):
        for pred_example, golden_example in zip(batch_pred_examples, batch_examples):
            pred_spo_list = self._del_duplicate(pred_example["spo_list"])
            golden_spo_list = golden_example["spo_list"]

            self.predict_count += len(pred_spo_list)
            self.recall_count += len(golden_spo_list)
            for pred_spo in pred_spo_list:
                if self._is_spo_in_list(pred_spo, golden_spo_list):
                    self.correct_count += 1

    def accumulate(self):
        precision_score = self.correct_count / self.predict_count if self.predict_count > 0 else 0.
        recall_score = self.correct_count / self.recall_count if self.recall_count > 0 else 0.
        f1_score = (2 * precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0.

        return precision_score, recall_score, f1_score

    def reset(self):
        self.correct_count = 0
        self.predict_count = 0
        self.recall_count = 0

    def name(self):
        return "SPOMetric"

    def _del_duplicate(self, spo_list):
        normalized_spo_list = []
        for spo in spo_list:
            if not self._is_spo_in_list(spo, normalized_spo_list):
                normalized_spo_list.append(spo)

        return normalized_spo_list

    def _is_spo_in_list(self, spo, spo_list):
        # check whether spo is in spo_list
        if spo in spo_list:
            return True
        return False


