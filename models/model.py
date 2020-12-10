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

import torch.nn as nn
import torch
import transformers
from models.layers import SimpleEncoder, SimpleMaxPoolDecoder, SubsampledBiLSTMEncoder, SimpleMaxPoolClassifier, SimpleSeqDecoder, get_bert, MaskedMaxPool, ConvolutionalSubsampledBiLSTMEncoder


class SLUModelBase(nn.Module):
    """Baseclass for SLU models"""
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class SLU(SLUModelBase):
    """Baseline SLU model"""
    def __init__(self, input_dim, encoder_dim, num_layers, num_classes):
        super().__init__()
        self.encoder = SimpleEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers)
        self.decoder = SimpleMaxPoolDecoder(input_dim=encoder_dim, hidden_dim=encoder_dim, num_classes=num_classes)

    def forward(self, feats, lengths):
        hiddens = self.encoder(feats, lengths)
        logits = self.decoder(hiddens, lengths)
        return logits


class SubsampledSLU(SLUModelBase):
    """Subsampled SLU model"""
    def __init__(self, input_dim, encoder_dim, num_layers, num_classes, decoder_hiddens=[]):
        super().__init__()
        self.encoder = SubsampledBiLSTMEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers)
        self.decoder = SimpleMaxPoolClassifier(input_dim=2*encoder_dim, num_classes=num_classes, hiddens=decoder_hiddens)

    def forward(self, feats, lengths):
        hiddens, lengths = self.encoder(feats, lengths)
        logits = self.decoder(hiddens, lengths)
        return logits


class Seq2Seq(SLUModelBase):
    """Baseline sequence 2 sequence implementation for SLU"""
    def __init__(self, input_dim, encoder_dim, num_layers, embedding_dim, vocab_size, num_heads):
        super().__init__()
        self.encoder = SubsampledBiLSTMEncoder(input_dim, encoder_dim, num_layers)
        self.decoder = SimpleSeqDecoder(vocab_size, embedding_dim, 2*encoder_dim, num_heads)

    def forward(self, feats, lengths, targets=None, training=False):
        hiddens, lengths = self.encoder(feats, lengths)
        if training:
            predictions = self.decoder(hiddens, hiddens, lengths, targets=targets, training=True)
        else:
            predictions = self.decoder(hiddens, hiddens, lengths, targets=None, training=False)
        return predictions


class BertNLU(nn.Module):
    """BERT NLU module"""
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        self.bert = get_bert(pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, num_classes)
        )

    def forward(self, input_text, text_lengths):
        batch_size = input_text.shape[0]
        max_seq_len = input_text.shape[1]
        mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
        mask = mask.long() # Convert to 0-1
        _, pooled_output = self.bert(input_ids=input_text, attention_mask=mask)
        logits = self.classifier(pooled_output)
        return logits


class JointModel(nn.Module):
    """JointModel which combines both modalities"""
    def __init__(self, input_dim, num_layers, num_classes, encoder_dim=None, bert_pretrained=True, bert_pretrained_model_name='bert-base-cased'):
        super().__init__()
        self.bert = get_bert(bert_pretrained, bert_pretrained_model_name)
        self.encoder_dim = encoder_dim
        if encoder_dim is None:
            self.speech_encoder = SubsampledBiLSTMEncoder(input_dim=input_dim, encoder_dim=self.bert.config.hidden_size//2, num_layers=num_layers)
        else:
            self.speech_encoder = SubsampledBiLSTMEncoder(input_dim=input_dim, encoder_dim=encoder_dim, num_layers=num_layers)
            self.aux_embedding = nn.Linear(2*encoder_dim, self.bert.config.hidden_size)
        self.maxpool = MaskedMaxPool()
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, audio_feats=None, audio_lengths=None, input_text=None, text_lengths=None, text_only=False):
        if text_only:
            return self.forward_text(input_text, text_lengths)
        outputs = {}
        if audio_feats is not None:
            hiddens, lengths = self.speech_encoder(audio_feats, audio_lengths)
            if self.encoder_dim is not None:
                hiddens = self.aux_embedding(hiddens)
            audio_embedding = self.maxpool(hiddens, lengths)
            audio_logits = self.classifier(audio_embedding)
            outputs['audio_embed'], outputs['audio_logits'] = audio_embedding, audio_logits

        if input_text is not None:
            batch_size = input_text.shape[0]
            max_seq_len = input_text.shape[1]
            attn_mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
            attn_mask = attn_mask.long() # Convert to 0-1
            _, text_embedding = self.bert(input_ids=input_text, attention_mask=attn_mask)
            text_logits = self.classifier(text_embedding)
            outputs['text_embed'], outputs['text_logits'] = text_embedding, text_logits

        return outputs

    def forward_text(self, input_text, text_lengths):
        outputs = {}
        batch_size = input_text.shape[0]
        max_seq_len = input_text.shape[1]
        attn_mask = torch.arange(max_seq_len, device=text_lengths.device)[None,:] < text_lengths[:,None]
        attn_mask = attn_mask.long() # Convert to 0-1
        _, text_embedding = self.bert(input_ids=input_text, attention_mask=attn_mask)
        text_logits = self.classifier(text_embedding)
        outputs['text_embed'], outputs['text_logits'] = text_embedding, text_logits
        return outputs
