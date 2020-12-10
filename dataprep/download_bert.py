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

import sys
import os
from transformers import BertModel, BertConfig, tokenization_bert
import urllib.request

if len(sys.argv) < 3:
    print("Usage: {} <name of BERT model> <path to save model>".format(sys.argv[0]))
    sys.exit(1)

model_name = sys.argv[1]
output_path = sys.argv[2]

# Part 1: download BERT model and save in model path
bert = BertModel.from_pretrained(model_name)

try:
    os.mkdir(output_path)
except FileExistsError:
    print("Warning, directory {} already exists.".format(output_path))

bert.save_pretrained(output_path)

# Part 2: download vocab file and save in model path
url = tokenization_bert.PRETRAINED_VOCAB_FILES_MAP["vocab_file"][model_name]
vocab_filename = os.path.join(output_path, "vocab.txt")
urllib.request.urlretrieve(url, vocab_filename)
