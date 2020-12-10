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

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import json


data_root = 'snips_slu'
df = pd.read_csv(os.path.join(data_root, 'data/complete.csv'))

new_paths = [os.path.join('wavs', 'audio', os.path.basename(path)) for path in df['path']]
df['path'] = new_paths

with open(os.path.join(data_root, 'data', 'intents.json'), 'r') as f:
    intent_label_dict = json.load(f)

# Extract only the intent from the semantics string. This is very hacky
intent_text_list = [semantics[12:].split("'")[0] for semantics in df['semantics']]
intent_labels_list = [intent_label_dict[intent_text] for intent_text in intent_text_list]

df['intent'] = intent_text_list
df['intent_label'] = intent_labels_list

# Split dataset into 80-10-10 train/dev/test
df_train, df_valtest = train_test_split(df, test_size=0.2, random_state=42)
df_val, df_test = train_test_split(df_valtest, test_size=0.5, random_state=42)

df_train.to_csv(os.path.join(data_root, 'data', 'train_data.csv'))
df_val.to_csv(os.path.join(data_root, 'data', 'valid_data.csv'))
df_test.to_csv(os.path.join(data_root, 'data', 'test_data.csv'))
