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

import torch
import torch.nn as nn
from torch.nn.utils import rnn
import numpy as np
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_layers, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size=input_dim, num_layers=num_layers, hidden_size=encoder_dim, batch_first=True, dropout=dropout)

    def forward(self, feats, lengths):
        '''
        Args:
            feats: Padded batch of utterances (batch_size, max_seq_len, input_dim)
            lengths: Lengths of utterances (batch_size,)
        '''
        out, _ = self.rnn(feats)
        return out


class ConvolutionalSubsampledBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_layers):
        super().__init__()
        lstms = [nn.LSTM(input_dim if i==0 else 2*encoder_dim, encoder_dim, 1, batch_first=True, bidirectional=True) for i in range(num_layers)]
        self.lstms = nn.ModuleList(lstms)
        self.subsample = Subsample()

    def forward(self, feats, lengths):
        x = feats
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                x, lengths = self.subsample(x, lengths)
            x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm.flatten_parameters()
            x, _ = lstm(x)
            x, _ = rnn.pad_packed_sequence(x, batch_first=True)
            x = F.relu(x)
        return x, lengths


class SubsampledBiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_layers):
        super().__init__()
        lstms = [nn.LSTM(input_dim if i==0 else 2*encoder_dim, encoder_dim, 1, batch_first=True, bidirectional=True) for i in range(num_layers)]
        self.lstms = nn.ModuleList(lstms)
        self.subsample = Subsample()

    def forward(self, feats, lengths):
        x = feats
        for i, lstm in enumerate(self.lstms):
            if i > 0:
                x, lengths = self.subsample(x, lengths)
            x = rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm.flatten_parameters()
            x, _ = lstm(x)
            x, _ = rnn.pad_packed_sequence(x, batch_first=True)
        return x, lengths


class Subsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, lengths):
        out = feats[:,::2]
        lengths = lengths // 2
        return out, lengths


class SimpleDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_dim, num_layers):
        super().__init__()
        self.attn = nn.MultiheadAttention()


class SimpleMaxPoolClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hiddens=[]):
        super().__init__()
        sizes = [input_dim] + hiddens
        classifier = [get_fc(in_size, out_size, True, 'relu') for in_size, out_size in zip(sizes[:1], sizes[1:])]
        classifier.append(get_fc(sizes[-1], num_classes, False, None))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, feats, lengths):
        max_seq_len = feats.shape[1]
        mask = torch.arange(max_seq_len, device=lengths.device)[None,:] < lengths[:,None]
        feats[~mask] = -np.inf
        feats = torch.max(feats, dim=1)[0]
        out = self.classifier(feats)
        return out


class SimpleMaxPoolDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, feats, lengths):
        out = self.dropout(feats)
        out, _ = self.rnn(feats)
        out = self.classifier(out)
        max_seq_len = feats.shape[1]
        mask = torch.arange(max_seq_len, device=lengths.device)[None,:] < lengths[:,None]
        out[~mask] = -np.inf
        out = torch.max(out, dim=1)[0]
        return out


class MaskedMaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feats, lengths):
        max_seq_len = feats.shape[1]
        mask = torch.arange(max_seq_len, device=lengths.device)[None,:] < lengths[:,None]
        feats[~mask] = -np.inf
        out = torch.max(feats, dim=1)[0]
        return out


class Attention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(model_dim, num_heads)

    def forward(self, query, key, value, lengths):
        key = key.permute(1,0,2) # T,N,D
        value = value.permute(1,0,2) # T,N,D
        query = query[None,:,:] # 1 x N x E
        max_seq_len = key.shape[0]
        mask = torch.arange(max_seq_len, device=lengths.device)[None,:] < lengths[:,None]
        attn_output, attn_weights = self.attention(query, key, value, key_padding_mask=~mask)
        return attn_output.squeeze(0), attn_weights


class SimpleSeqDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm1 = nn.LSTMCell(embedding_dim+hidden_dim, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim, num_heads)
        self.classifier = nn.Linear(hidden_dim+hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, keys, values, lengths, targets=None, training=False):
        batch_size = keys.shape[0]
        if training:
            max_len = targets.shape[1]
            embeddings = self.embedding(targets)
        else:
            max_len = 10
        predictions = []
        hidden_states = [(torch.zeros(batch_size, self.hidden_dim, device=keys.device),
            torch.zeros(batch_size, self.hidden_dim, device=keys.device)) for _ in range(2)]
        prediction = torch.zeros(batch_size, 1).to(keys.device)
        TEACHER_FORCE_P = 0.8
        for i in range(max_len-1):
            if training:
                p = np.random.uniform()
                if p < TEACHER_FORCE_P:
                    char_embed = embeddings[:,i,:]
                else:
                    char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            query = hidden_states[1][0]
            context, _ = self.attention(query, keys, values, lengths)
            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]
            prediction = self.classifier(torch.cat([output, context], dim=1))
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)


def get_bert(pretrained=True, pretrained_model_name='bert-base-cased'):
    """Initialize a pretrained BERT model"""
    if pretrained:
        bert = BertModel.from_pretrained(pretrained_model_name)
    else:
        configuration = BertConfig.from_pretrained(pretrained_model_name)
        bert = BertModel(configuration)
    return bert


def get_fc(input_size, output_size, batch_normalization=False, activation=None):
    layers = [nn.Linear(input_size, output_size)]
    if batch_normalization:
        layers.append(nn.BatchNorm1d(output_size))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)
