# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')


def get_genotype(model):
    def _parse(weights, weights2=None):
        gene = []
        n = 2
        start = 0
        for i in range(model._steps):
            end = start + n
            W = weights[start:end].copy()
            if model._method == "PC-DARTS":
                W2 = weights2[start:end].copy()
                for j in range(n):
                    W[j, :] = W[j, :] * W2[j]
            edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != model._primitives.index('none')))[:2]
            for j in edges:
                k_best = None
                for k in range(len(W[j])):
                    if k != model._primitives.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((model._primitives[k_best], j))
            start = end
            n += 1
        return gene

    weightsr2 = None
    weightsn2 = None
    if model._method == "PC-DARTS":
        n = 3
        start = 2
        weightsr2 = fluid.layers.softmax(model.betas_reduce[0:2])
        weightsn2 = fluid.layers.softmax(model.betas_normal[0:2])
        for i in range(model._steps - 1):
            end = start + n
            tw2 = fluid.layers.softmax(model.betas_reduce[start:end])
            tn2 = fluid.layers.softmax(model.betas_normal[start:end])
            start = end
            n += 1
            weightsr2 = fluid.layers.concat([weightsr2, tw2])
            weightsn2 = fluid.layers.concat([weightsn2, tn2])
        weightsr2 = weightsr2.numpy()
        weightsn2 = weightsn2.numpy()

    gene_normal = _parse(
        fluid.layers.softmax(model.alphas_normal).numpy(), weightsn2)
    gene_reduce = _parse(
        fluid.layers.softmax(model.alphas_reduce).numpy(), weightsr2)

    concat = range(2 + model._steps - model._multiplier, model._steps + 2)
    genotype = Genotype(
        normal=gene_normal,
        normal_concat=concat,
        reduce=gene_reduce,
        reduce_concat=concat)
    return genotype
