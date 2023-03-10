# -*- coding:utf8 -*-
from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
# from torchtext.data import Iterator, BucketIterator
# from torchtext import data
import torch


def load_iters(batch_size=32, device="cpu", data_path='data', vectors=None, limit=100000):
    TEXT = data.Field(lower=True, batch_first=True, include_lengths=True)
    LABEL = data.LabelField(batch_first=True)
    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL)}

    train_data, test_data = data.TabularDataset.splits(
        path=data_path,
        train='train.jsonl',
        test='test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )
    train_data = data.Dataset(train_data.examples[:limit], train_data.fields)
    dev_data = test_data
    print(f'using {len(train_data)} train data...')


    if vectors is not None:
        TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data, max_size=50000)
    LABEL.build_vocab(train_data.label)
    train_iter, dev_iter = BucketIterator.splits(
        (train_data, dev_data),
        batch_sizes=(batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True
    )

    test_iter = Iterator(test_data, batch_size=batch_size, device=device, sort=False, sort_within_batch=False,
                         repeat=False, shuffle=False)
    return train_iter, dev_iter, test_iter, TEXT, LABEL
