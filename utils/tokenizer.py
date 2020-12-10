# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import numpy as np

class Tokenizer:
    def __init__(self):
        self.words = set()
        self.PAD_TOK = '<PAD>'
        self.SOS_TOK = '<SOS>'
        self.EOS_TOK = '<EOS>'

    def add_words(self, words):
        self.words = self.words.union(set(words))

    def make_dicts(self):
        word_list = list(self.words)
        word_list = [self.PAD_TOK, self.SOS_TOK, self.EOS_TOK] + word_list
        self._word2idx = dict(zip(word_list, range(len(word_list))))
        self._idx2word = dict(zip(range(len(word_list)), word_list))

    def import_json(self, word2idx_json):
        with open(word2idx_json, 'r') as f:
            self._word2idx = json.load(f)
        self._idx2word = {v:k for k,v in self._word2idx.items()}

    def export_json(self, word2idx_json):
        with open(word2idx_json, 'w', encoding='utf-8') as f:
            json.dump(self._word2idx, f, ensure_ascii=False, indent=4)

    def tokenize(self, words):
        words = [self.SOS_TOK] + words + [self.EOS_TOK]
        return np.array([self._word2idx[word] for word in words])

    def decode(self, indices):
        words = [self._idx2word[idx] for idx in indices]
        return ', '.join(words)

    @property
    def vocab_size(self):
        return len(self._word2idx)
