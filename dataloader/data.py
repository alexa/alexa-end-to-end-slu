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
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
from utils.tokenizer import Tokenizer
from transformers import BertTokenizer
import torchaudio
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os


class BaseDataset(Dataset):

    def load_audio(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(self.data_root, df_row['path'])
        waveform, sample_rate = torchaudio.load(filename)
        fbank_feats = torchaudio.compliance.kaldi.mfcc(waveform, num_ceps=40, num_mel_bins=80)
        intent = df_row['intent_label']
        encoding = self.bert_tokenizer.encode_plus(
            df_row['transcription'],
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt'
            )
        return fbank_feats, intent, encoding, df_row['transcription']

    def get_dict(self, fbank_feats, intent, encoding, transcription, suffix=''):
        ret_dict = {'feats':fbank_feats, 'length':fbank_feats.shape[0] ,'label':intent, 'encoded_text':encoding['input_ids'].flatten(),
                'text_length':encoding['input_ids'].flatten().shape[0], 'raw_text':transcription}
        ret_dict = {k+suffix:v for k,v in ret_dict.items()}
        return ret_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raise NotImplementedError

    def labels_list(self):
        raise NotImplementedError


def triplet_getter(self, idx):
    fbank_feats, intent, encoding, transcription = self.load_audio(idx)

    while True:
        pos_idx = np.random.choice(self.label2idx[intent])
        if pos_idx != idx:
            break

    neg_label = np.random.choice(list(self.labels_set - set([intent])))
    neg_idx = np.random.choice(self.label2idx[neg_label])

    fbank_feats2, intent2, encoding2, transcription2 = self.load_audio(pos_idx)
    fbank_feats3, intent3, encoding3, transcription3 = self.load_audio(neg_idx)

    dict1 = self.get_dict(fbank_feats, intent, encoding, transcription)
    dict2 = self.get_dict(fbank_feats2, intent2, encoding2, transcription2, suffix='2')
    dict3 = self.get_dict(fbank_feats3, intent3, encoding3, transcription3, suffix='3')
    return {**dict1, **dict2, **dict3}


class BaseFluentSpeechDataset(BaseDataset):
    '''Baseclass for the Fluent Speech Commands dataset'''
    def __init__(self, data_root, split='train', intent_encoder=None, pretrained_model_name='bert-base-cased'):
        assert split in ['train', 'test', 'valid'], 'Invalid split'

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(self.data_root, 'data/', '{}_data.csv'.format(split)))
        self.df['intent'] = self.df[['action', 'object', 'location']].apply('-'.join, axis=1)

        if intent_encoder is None:
            intent_encoder = preprocessing.LabelEncoder()
            intent_encoder.fit(self.df['intent'])
        self.intent_encoder = intent_encoder
        self.df['intent_label'] = intent_encoder.transform(self.df['intent'])
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        self.labels_set = set(self.df['intent_label'])
        self.label2idx = {}
        for label in self.labels_set:
            idx = np.where(self.df['intent_label'] == label)[0]
            self.label2idx[label] = idx

    def labels_list(self):
        return self.intent_encoder.classes_


class BaseSnipsSLUDataset(BaseDataset):
    '''Baseclass for the Snips SLU dataset'''
    def __init__(self, data_root, split='train', intent_encoder=None, pretrained_model_name='bert-base-cased'):

        assert split in ['train', 'test', 'valid'], 'Invalid split'

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(self.data_root, 'data/', '{}_data.csv'.format(split)))
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

        self.labels_set = set(self.df['intent_label'])
        self.label2idx = {}
        for label in self.labels_set:
            idx = np.where(self.df['intent_label'] == label)[0]
            self.label2idx[label] = idx

    def labels_list(self):
        with open(os.path.join(self.data_root, 'data', 'intents.json'), 'r') as f:
            import json
            intent_label_dict = json.load(f)
            intent_labels = list(intent_label_dict.keys())
            return intent_labels


class FluentSpeechDataset(BaseFluentSpeechDataset):

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fbank_feats, intent, encoding, transcription = self.load_audio(idx)
        return self.get_dict(fbank_feats, intent, encoding, transcription)


class SnipsSLUDataset(BaseSnipsSLUDataset):
    def __init__(self, data_root, split='train', pretrained_model_name='bert-base-cased'):

        assert split in ['train', 'test', 'valid'], 'Invalid split'

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(self.data_root, 'data/', '{}_data.csv'.format(split)))
        self.bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fbank_feats, intent, encoding, transcription = self.load_audio(idx)
        return self.get_dict(fbank_feats, intent, encoding, transcription)


class FluentSpeechTripletDataset(BaseFluentSpeechDataset):
    def __getitem__(self, idx):
        return triplet_getter(self, idx)


class SnipsSLUTripletDataset(BaseSnipsSLUDataset):
    def __getitem__(self, idx):
        return triplet_getter(self, idx)


class FluentSeq2SeqDataset(BaseFluentSpeechDataset):
    def __init__(self, data_root, split='train', vocab_json=None):

        assert split in ['train', 'test', 'valid'], 'Invalid split'

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(self.data_root, 'data/', '{}_data.csv'.format(split)))
        self.lang = Tokenizer()
        if vocab_json is None:
            self.lang.add_words(self.df['action'])
            self.lang.add_words(self.df['object'])
            self.lang.add_words(self.df['location'])
            self.lang.make_dicts()
            self.vocab_json = 'word2idx.json'
            self.lang.export_json(self.vocab_json)
        else:
            self.vocab_json = vocab_json
            self.lang.import_json(vocab_json)

    @property
    def vocab_size(self):
        return self.lang._word2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(self.data_root, df_row['path'])
        waveform, sample_rate = torchaudio.load(filename)
        fbank_feats = torchaudio.compliance.kaldi.mfcc(waveform, num_ceps=40, num_mel_bins=80)
        output = self.lang.tokenize([df_row['action'], df_row['object'], df_row['location']])
        return {'feats':fbank_feats, 'length':fbank_feats.shape[0], 'output':output, 'output_length': output.shape[0]}


def default_collate_classifier(inputs):
    '''
    Pads and collates into a batch for training
    Returns:
        A dictionary containing
        'feats': (B,*,D) where * denotes the maximum sequence length
        'length': (B,) length of each utterance
        'label': (B,) label of each utterance
    '''
    feats = [data['feats'] for data in inputs]
    labels = [data['label'] for data in inputs]
    lengths = [data['length'] for data in inputs]
    raw_text = [data['raw_text'] for data in inputs]
    encoded_text = [data['encoded_text'] for data in inputs]
    text_lengths = [data['text_length'] for data in inputs]
    padded_feats = rnn.pad_sequence(feats, batch_first=True)
    padded_text = rnn.pad_sequence(encoded_text, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)
    return {'feats':padded_feats, 'length':lengths, 'label':labels, 'encoded_text':padded_text, 'text_length':text_lengths, 'raw_text':raw_text}


def default_collate_seq2seq(inputs):
    '''
    Pads and collates into batch for Seq2Seq training
    Returns:
        A dictionary containing
        'feats': (B,*,D) where * denotes the maximum sequence length
        'length': (B,) length of each utterance
        'output': (B,*) output tokens where * denotes the maximum sequence length
        'output_length': (B,) length of each output
    '''
    feats = [data['feats'] for data in inputs]
    outputs = [torch.tensor(data['output'], dtype=torch.long) for data in inputs]
    lengths = [data['length'] for data in inputs]
    output_lengths = [data['output_length'] for data in inputs]
    padded_feats = rnn.pad_sequence(feats, batch_first=True)
    padded_outputs = rnn.pad_sequence(outputs, batch_first=True)
    lengths = torch.tensor(lengths, dtype=torch.long)
    output_lengths = torch.tensor(output_lengths, dtype=torch.long)
    return {'feats':padded_feats, 'length':lengths, 'output':padded_outputs, 'output_length':output_lengths}


def default_collate_pairwise(inputs):
    '''
    Pads and collates into a batch for training
    Returns:
        A dictionary containing
        'feats': (B,*,D) where * denotes the maximum sequence length
        'length': (B,) length of each utterance
        'label': (B,) label of each utterance
    '''
    feats = [data['feats'] for data in inputs]
    labels = [data['label'] for data in inputs]
    lengths = [data['length'] for data in inputs]
    raw_text = [data['raw_text'] for data in inputs]
    encoded_text = [data['encoded_text'] for data in inputs]
    text_lengths = [data['text_length'] for data in inputs]
    padded_feats = rnn.pad_sequence(feats, batch_first=True)
    padded_text = rnn.pad_sequence(encoded_text, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)

    encoded_text2 = [data['encoded_text2'] for data in inputs]
    text_lengths2 = [data['text_length2'] for data in inputs]
    raw_text2 = [data['raw_text2'] for data in inputs]
    padded_text2 = rnn.pad_sequence(encoded_text2, batch_first=True)
    text_lengths2 = torch.tensor(text_lengths2, dtype=torch.long)

    targets = [data['target'] for data in inputs]
    targets = torch.tensor(targets, dtype=torch.long)
    return {'feats':padded_feats, 'length':lengths, 'label':labels, 'encoded_text':padded_text, 'text_length':text_lengths, 'raw_text':raw_text,
            'encoded_text2':padded_text2, 'text_length2':text_lengths2, 'raw_text2':raw_text2, 'target':targets}


def default_collate_triplet(inputs):
    '''
    Pads and collates into a batch for training
    Returns:
        A dictionary containing
        'feats': (B,*,D) where * denotes the maximum sequence length
        'length': (B,) length of each utterance
        'label': (B,) label of each utterance
    '''
    feats = [data['feats'] for data in inputs]
    labels = [data['label'] for data in inputs]
    lengths = [data['length'] for data in inputs]
    raw_text = [data['raw_text'] for data in inputs]
    encoded_text = [data['encoded_text'] for data in inputs]
    text_lengths = [data['text_length'] for data in inputs]
    padded_feats = rnn.pad_sequence(feats, batch_first=True)
    padded_text = rnn.pad_sequence(encoded_text, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)

    encoded_text2 = [data['encoded_text2'] for data in inputs]
    text_lengths2 = [data['text_length2'] for data in inputs]
    raw_text2 = [data['raw_text2'] for data in inputs]
    padded_text2 = rnn.pad_sequence(encoded_text2, batch_first=True)
    text_lengths2 = torch.tensor(text_lengths2, dtype=torch.long)

    encoded_text3 = [data['encoded_text3'] for data in inputs]
    text_lengths3 = [data['text_length3'] for data in inputs]
    raw_text3 = [data['raw_text3'] for data in inputs]
    padded_text3 = rnn.pad_sequence(encoded_text3, batch_first=True)
    text_lengths3 = torch.tensor(text_lengths3, dtype=torch.long)

    return {'feats':padded_feats, 'length':lengths, 'label':labels, 'encoded_text':padded_text, 'text_length':text_lengths, 'raw_text':raw_text,
            'encoded_text2':padded_text2, 'text_length2':text_lengths2, 'raw_text2':raw_text2,
            'encoded_text3':padded_text3, 'text_length3':text_lengths3, 'raw_text3':raw_text3}


def get_dataloaders(data_root, batch_size, dataset='fsc', num_workers=0, *args, **kwargs):
    if dataset == 'fsc':
        train_dataset = FluentSpeechDataset(data_root, 'train', *args, **kwargs)
        val_dataset = FluentSpeechDataset(data_root, 'valid', train_dataset.intent_encoder, *args, **kwargs)
        test_dataset = FluentSpeechDataset(data_root, 'test', train_dataset.intent_encoder, *args, **kwargs)
    elif dataset == 'snips':
        train_dataset = SnipsSLUDataset(data_root, 'train', *args, **kwargs)
        val_dataset = SnipsSLUDataset(data_root, 'valid', *args, **kwargs)
        test_dataset = SnipsSLUDataset(data_root, 'test', *args, **kwargs)
    else:
        raise ValueError('Invalid dataset')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_collate_classifier, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_seq2seq_dataloaders(data_root, batch_size, vocab_json=None, num_workers=0, *args, **kwargs):
    train_dataset = FluentSeq2SeqDataset(data_root, 'train', vocab_json, *args, **kwargs)
    if vocab_json is None:
        vocab_json = train_dataset.vocab_json
    val_dataset = FluentSeq2SeqDataset(data_root, 'valid', vocab_json, *args, **kwargs)
    test_dataset = FluentSeq2SeqDataset(data_root, 'test', vocab_json, *args, **kwargs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_collate_seq2seq, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=default_collate_seq2seq, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_collate_seq2seq, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_pairwise_dataloaders(data_root, batch_size, dataset='fsc', num_workers=0, *args, **kwargs):
    if dataset == 'fsc':
        train_dataset = FluentSpeechPairwiseDataset(data_root, 'train', *args, **kwargs)
        val_dataset = FluentSpeechPairwiseDataset(data_root, 'valid', train_dataset.intent_encoder, *args, **kwargs)
        test_dataset = FluentSpeechPairwiseDataset(data_root, 'test', train_dataset.intent_encoder, *args, **kwargs)
    elif dataset == 'snips':
        train_dataset = SnipsSLUPairwiseDataset(data_root, 'train', *args, **kwargs)
        val_dataset = SnipsSLUPairwiseDataset(data_root, 'valid', *args, **kwargs)
        test_dataset = SnipsSLUPairwiseDataset(data_root, 'test', *args, **kwargs)
    else:
        raise ValueError("No valid dataset selected!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_collate_pairwise, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=default_collate_pairwise, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_collate_pairwise, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_triplet_dataloaders(data_root, batch_size, dataset='fsc', num_workers=0, *args, **kwargs):
    if dataset == 'fsc':
        train_dataset = FluentSpeechTripletDataset(data_root, 'train', *args, **kwargs)
        val_dataset = FluentSpeechTripletDataset(data_root, 'valid', train_dataset.intent_encoder, *args, **kwargs)
        test_dataset = FluentSpeechTripletDataset(data_root, 'test', train_dataset.intent_encoder, *args, **kwargs)
    elif dataset == 'snips':
        train_dataset = SnipsSLUTripletDataset(data_root, 'train', *args, **kwargs)
        val_dataset = SnipsSLUTripletDataset(data_root, 'valid', *args, **kwargs)
        test_dataset = SnipsSLUTripletDataset(data_root, 'test', *args, **kwargs)
    else:
        raise ValueError("No valid dataset selected!")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=default_collate_triplet, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=default_collate_triplet, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=default_collate_triplet, num_workers=num_workers)

    return train_loader, val_loader, test_loader
