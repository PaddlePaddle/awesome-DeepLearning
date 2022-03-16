# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import numpy as np
from PIL import Image
import decord as de
import copy
import json
from ..registry import PIPELINES

from paddlenlp.transformers import BertTokenizer


@PIPELINES.register()
class FeaturePadding(object):
    """
    Padding feature to target shape.
    """
    def __init__(self, max_region_num=36, max_action_num=5):
        self.max_region_num = max_region_num
        self.max_action_num = max_action_num

    def __call__(self, results):
        """
        Padding feature.
        """
        pack_feature = results['feature']
        tokenizer = results['tokenizer']
        image_feature_wp, image_target_wp, image_location_wp, \
                num_boxes,  image_h, image_w, image_id, caption, \
                action_feature_wp, action_target_wp, num_actions = pack_feature

        image_feature = np.zeros((self.max_region_num, 2048), dtype=np.float32)
        image_target = np.zeros((self.max_region_num, 1601), dtype=np.float32)
        image_location = np.zeros((self.max_region_num, 5), dtype=np.float32)

        action_feature = np.zeros((self.max_action_num, 2048), dtype=np.float32)
        action_target = np.zeros((self.max_action_num, ), dtype=np.int64)

        num_boxes = int(num_boxes)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes, :4] = image_location_wp

        image_location[:, 4] = (image_location[:, 3] - image_location[:, 1]) * (
            image_location[:, 2] - image_location[:, 0]) / (float(image_w) *
                                                            float(image_h))

        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)

        image_feature = copy.deepcopy(image_feature)
        image_target = copy.deepcopy(image_target)

        num_actions = int(num_actions)
        action_feature[:num_actions] = action_feature_wp
        action_target[:num_actions] = action_target_wp
        action_feature = copy.deepcopy(action_feature)
        action_target = copy.deepcopy(action_target)

        results = dict(image_feat=image_feature,
                       image_target=image_target,
                       caption=caption,
                       image_loc=image_location,
                       num_boxes=int(num_boxes),
                       action_feat=action_feature,
                       action_target=action_target,
                       num_actions=int(num_actions),
                       tokenizer=tokenizer)
        return results


@PIPELINES.register()
class RandomCap(object):
    def __init__(self, caption_path):
        """
        Random Caption for NSP task
        """
        self.caption_path = caption_path

    def select_caption(self, caption):
        captions = caption.split('!')
        rind = random.randint(0, len(captions) - 1)
        caption = captions[rind]
        return caption

    def get_random_caption(self, all_captions):
        num_caps = len(all_captions)
        rand_doc_idx = random.randint(0, num_caps - 1)
        caption = all_captions[rand_doc_idx]
        caption = self.select_caption(caption)
        return caption

    def random_cap(self, caption, all_captions):
        if random.random() > 0.5:
            label = 0
        else:
            caption = self.get_random_caption(all_captions)
            label = 1
        return caption, label

    def __call__(self, results):
        caption = results['caption']
        all_captions = list(json.load(open(self.caption_path, 'r')))
        caption = self.select_caption(caption)
        caption, label = self.random_cap(caption, all_captions)
        results['caption'] = caption
        results['is_next'] = label
        return results


@PIPELINES.register()
class Tokenize(object):
    def __init__(self, ):
        """
        Tokenize caption
        """
        pass

    def __call__(self, results):
        caption = results['caption']
        tokenizer = results['tokenizer']
        tokens_caption = tokenizer.tokenize(caption)
        results['caption'] = tokens_caption
        return results


@PIPELINES.register()
class RandomMask(object):
    def __init__(self,
                 max_seq_length=36,
                 max_action_length=5,
                 max_region_length=36):
        self.max_seq_length = max_seq_length
        self.max_action_length = max_action_length
        self.max_region_length = max_region_length

    def get_image_global_feature(self, image_feat, image_loc, image_mask):
        g_image_feat = np.sum(image_feat, axis=0) / np.sum(
            image_mask, axis=0, keepdims=True)
        image_feat = np.concatenate(
            [np.expand_dims(g_image_feat, axis=0), image_feat],
            axis=0).astype("float32")

        g_image_loc = np.array([0, 0, 1, 1, 1]).astype("float32")
        image_loc = np.concatenate(
            [np.expand_dims(g_image_loc, axis=0), image_loc], axis=0)

        g_image_mask = np.array([1])
        image_mask = np.concatenate([g_image_mask, image_mask], axis=0)

        return image_feat, image_loc, image_mask

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length.
        This is a simple heuristic which will always truncate the longer sequence
        one token at a time. This makes more sense than truncating an equal percent
        of tokens from each, since if one sequence is very short then each token
        that's truncated likely contains more information than a longer sequence.
        """
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break
            tokens_b.pop()

    def random_word(self, tokens, tokenizer):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        Args:
            tokens: list of str, tokenized sentence.
            tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        Return:
            (list of str, list of int), masked tokens and related labels for LM prediction
        """
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability

            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = "[MASK]"

                # 10% randomly change token to random token
                elif prob < 0.9:
                    #tok = random.choice(list(tokenizer.vocab.items()))[0]
                    tok = tokenizer.vocab.idx_to_token[random.randint(
                        0,
                        tokenizer.vocab_size,
                    )]
                    tokens[i] = tok

                # rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                try:
                    output_label.append(tokenizer.vocab[token])
                except KeyError:
                    # For unknown words (should not occur with BPE vocab)
                    output_label.append(tokenizer.vocab["[UNK]"])
                    print(
                        "Cannot find token '{}' in vocab. Using [UNK] insetad".
                        format(token))
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return tokens, output_label

    def random_region(self, image_feat, image_loc, num_boxes):
        output_label = []

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0

                # rest 20% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label

    def random_action(self, action_feat, action_target, num_actions):
        output_label = []

        for i in range(num_actions):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 90% randomly change token to mask token
                if prob < 0.9:
                    action_feat[i] = 0

                # rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(action_target[i])
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return action_feat, output_label

    def __call__(self, results):
        caption = results['caption']
        tokenizer = results['tokenizer']
        image_feat = results['image_feat']
        image_loc = results['image_loc']
        num_boxes = results['num_boxes']
        action_feat = results['action_feat']
        action_target = results['action_target']
        num_actions = results['num_actions']
        is_next = results['is_next']
        image_target = results['image_target']

        self._truncate_seq_pair(caption, self.max_seq_length - 2)
        caption, caption_label = self.random_word(caption, tokenizer)

        image_feat, image_loc, image_label = self.random_region(
            image_feat, image_loc, num_boxes)
        action_feat, action_label = self.random_action(action_feat,
                                                       action_target,
                                                       num_actions)

        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = [-1] + caption_label + [-1]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        tokens = []
        segment_ids = []

        tokens.append("[CLS]")
        segment_ids.append(0)

        for token in caption:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * (len(input_ids))
        image_mask = [1] * (num_boxes)
        action_mask = [1] * (num_actions)

        # Zero-pad up to the visual sequence length.
        while len(image_mask) < self.max_region_length:
            image_mask.append(0)
            image_label.append(-1)
        while len(action_mask) < self.max_action_length:
            action_mask.append(0)
            action_label.append(-1)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(lm_label_ids) == self.max_seq_length
        assert len(image_mask) == self.max_region_length
        assert len(image_label) == self.max_region_length
        assert len(action_mask) == self.max_action_length
        assert len(action_label) == self.max_action_length

        image_feat, image_loc, image_mask = self.get_image_global_feature(
            image_feat, image_loc, np.array(image_mask))
        features = [
            np.array(input_ids),
            action_feat,
            image_feat,
            image_loc,
            np.array(segment_ids),
            np.array(input_mask),
            image_mask,
            np.array(action_mask),
            np.array(lm_label_ids),
            np.array(action_label),
            np.array(is_next),
            np.array(image_label),
            image_target,
        ]
        results['features'] = features
        return results
