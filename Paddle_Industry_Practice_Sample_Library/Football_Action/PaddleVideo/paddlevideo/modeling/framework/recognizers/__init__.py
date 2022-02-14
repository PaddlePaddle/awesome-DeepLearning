# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from .base import BaseRecognizer
from .recognizer1d import Recognizer1D
from .recognizer2d import Recognizer2D
from .recognizer3d import Recognizer3D
from .recognizer_transformer import RecognizerTransformer
from .recognizer_gcn import RecognizerGCN
from .recognizerMRI import RecognizerMRI
from .recognizer3dMRI import Recognizer3DMRI
from .recognizer_transformer_MRI import RecognizerTransformer_MRI

__all__ = [
    'BaseRecognizer', 'Recognizer1D', 'Recognizer2D', 'Recognizer3D',
    'RecognizerTransformer', 'RecognizerGCN', 'RecognizerMRI',
    'Recognizer3DMRI', 'RecognizerTransformer_MRI'
]
