# -*- coding:utf8 -*-
from torchtext.legacy.data import Iterator, BucketIterator
from torchtext.legacy import data
# from torchtext.data import Iterator, BucketIterator
# from torchtext import data
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,SGD
import os
import sys
from models import RNN
import wandb
import random
import numpy as np
import json
import os
import argparse
from bilevel_tools.meta import MetaSGD
from bilevel_tools.tbtools import AverageMeter
import torch.nn.functional as F
import bilevel_tools.loss_utils as loss_utils
import math
import time
import datetime
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HIDDEN_SIZE = 300  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
DROPOUT_RATE = 0.5
LAYER_NUM = 1
EMBEDDING_SIZE = 100
vectors = None
freeze = False


def construct_outer_subloader(train_data, indices = None, idx_to_order=None):
    if indices is None:
        num_use_samples_inner=len(train_data.examples)
        indices = np.random.choice(list(range(num_use_samples_inner)), args.num_use_samples_outer, replace=False)
    else:
        indices = [idx_to_order[idx] for idx in indices]
    dev_data = data.Dataset([train_data.examples[ix] for ix in indices], train_data.fields)
    subset_iter= BucketIterator.splits(
        (dev_data,),
        batch_sizes=(args.backward_batch_size,),
        device='cuda',
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=True,
    )
    return subset_iter[0]


def load_iters( batch_size=32,backward_batch_size=1000, device="cpu", gold_data_path='data', syn_data_path='data', vectors=None, use_tree=False, num_use_samples_inner=100, num_use_samples_outer=100, shuffle_train=True):
    TEXT = data.Field(batch_first=True, include_lengths=True, lower=True)
    LABEL = data.LabelField(batch_first=True)
    INDEX = data.RawField()
    fields = {'text': ('text', TEXT),
              'label': ('label', LABEL),
              'idx': ('idx', INDEX)}

    train_data, _ = data.TabularDataset.splits(
        path=syn_data_path,
        train='train.jsonl',
        test='train.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )
    traindataset = train_data.examples[:num_use_samples_inner]
    random.shuffle(traindataset)
    train_data = data.Dataset(traindataset, train_data.fields)

    dev_data, test_data = data.TabularDataset.splits(
        path=gold_data_path,
        validation='train.jsonl',
        test='test.jsonl',
        format='json',
        fields=fields,
        filter_pred=lambda ex: ex.label != '-'  # filter the example which label is '-'(means unlabeled)
    )

    if args.use_dev_outer:
        dev_data_all = data.Dataset(dev_data.examples, dev_data.fields)
        dev_data = data.Dataset(dev_data.examples[:num_use_samples_outer], dev_data.fields)
    else:
        if args.subset_outer:
            indices = np.random.choice(list(range(num_use_samples_inner)), num_use_samples_outer, replace=False)
            dev_data = data.Dataset([train_data.examples[ix] for ix in indices], train_data.fields)
        else:
            dev_data=train_data
        dev_data_all=train_data


    if vectors is not None:
        TEXT.build_vocab(train_data, vectors=vectors, unk_init=torch.Tensor.normal_)
    else:
        TEXT.build_vocab(train_data, max_size=50000)
    LABEL.build_vocab(train_data)

    train_iter, train_iter_backward, dev_iter = BucketIterator.splits(
        (train_data, train_data, dev_data),
        batch_sizes=(batch_size, backward_batch_size, batch_size),
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        repeat=False,
        shuffle=shuffle_train,
    )

    test_iter = Iterator(test_data,
                         batch_size=batch_size,
                         device=device,
                         sort=False,
                         sort_within_batch=False,
                         repeat=False,
                         shuffle=False)

    return train_iter, train_iter_backward, dev_iter, test_iter, TEXT, LABEL, train_data, dev_data_all



def set_seed(seed = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(model, data_iter, name, epoch=None):
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    all_labels = []
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            (inputs, lens), labels = batch.text, batch.label
            output = model(inputs, lens)
            labels = batch.label
            all_labels.append(labels)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != batch.label).sum().item()

    acc = correct_num / (correct_num + err_num)
    if epoch is not None:
        tqdm.write(
            "Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, acc, total_loss))
    else:
        tqdm.write(
            "%s Acc: %.3f, Loss %.3f" % (name, acc, total_loss))
    all_labels = torch.cat(all_labels)
    print(f"num of zeros: {torch.sum(all_labels == 0)}")
    print(f"num of ones: {torch.sum(all_labels == 1)}")
    return acc, total_loss/len(data_iter)


def train(model, train_iter, dev_iter, loss_func, optimizer, epochs, patience=5, clip=5):
    best_model = copy.deepcopy(model)
    best_acc = -1
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_iter):
            (inputs, lens), labels = batch.text, batch.label
            labels = batch.label

            model.zero_grad()
            output = model(inputs, lens)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        tqdm.write("Epoch: %d, Train Loss: %d" % (epoch + 1, total_loss))

        acc = eval(model, dev_iter, "Dev", epoch)
        wandb.log({"loss": total_loss, "val_acc": acc})

        if acc<best_acc:
            patience_counter +=1
        else:
            best_acc = acc
            patience_counter = 0
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), 'best_model.ckpt')
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break
    return best_model

def train_to_converge(model, train_iter, theta, epoch_converge, inner_obj):
    model_copy = copy.deepcopy(model)
    if args.optim =='Adam':
        optimizer = Adam(model_copy.parameters(), lr=args.inner_lr)
    elif args.optim =='SGD':
        optimizer = SGD(model_copy.parameters(), lr=args.inner_lr, momentum=0.9)
    losses = AverageMeter("Loss", ":.3f")
    model_weights_cache = []
    opt_checkpoints_cache = []
    diverged = False
    for epoch in range(epoch_converge):
        model_copy.train()
        top1 = AverageMeter("OuterAcc@1", ":6.2f")
        for batch in tqdm(train_iter):
        # for batch in train_iter:
            (inputs, lens), labels = batch.text, batch.label
            labels = batch.label
            model_copy.zero_grad()
            output = model_copy(inputs, lens)
            if inner_obj == "ce":
                if not args.normalize:
                    loss = torch.mean(F.cross_entropy(output, labels, reduction='none').flatten()*theta[batch.idx])
                else:
                    loss = torch.sum(F.cross_entropy(output, labels, reduction='none').flatten()*theta[batch.idx])/torch.sum(theta[batch.idx])
            elif inner_obj=='kl':
                one_hot = torch.zeros(len(labels),len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss_vec = torch.mean(F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)), dim=1)
                loss = torch.mean(loss_vec*theta[batch.idx])
            acc = loss_utils.accuracy(output, labels)
            top1.update(acc, labels.size(0))
            losses.update(loss.item(), labels.size(0))
            loss.backward()
            optimizer.step()
    opt_checkpoints_cache.append(optimizer.state_dict())
    model_weights_cache.append(copy.deepcopy(model_copy.state_dict()))
    if math.isnan(loss.item()):
        diverged = True
    return model_copy, losses.avg, top1.avg, model_weights_cache, opt_checkpoints_cache, diverged

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def get_grad_weights_on_valid(model, val_iter, theta):
    grad_weights_on_full_train = []
    losses = AverageMeter("OuterLoss", ":.3f")
    top1 = AverageMeter("OuterAcc@1", ":6.2f")
    for batch_idx, batch in enumerate(val_iter):
        theta_batch = theta[batch.idx]
        (inputs, lens), labels = batch.text, batch.label
        labels = batch.label
        output = model(inputs, lens)
        if args.use_dev_outer:
            loss = loss_func(output, labels)
        else:
            if args.hard:
                val, idx = torch.topk(theta, int(args.threshold * len(theta)))
                subnet = (theta_batch >= val[-1]).float()
                selection = torch.nonzero(subnet.squeeze()).flatten()
            if args.outer_obj == "entropy":
                loss = - torch.mul(F.softmax(output), F.log_softmax(output))
            elif args.outer_obj == "mae":
                outputvar = output[:, labels]
                loss = (1. - outputvar)
            elif args.outer_obj == "kl":
                one_hot = torch.zeros(len(labels), len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot))
            else:
                one_hot = torch.zeros(len(labels), len(LABEL.vocab.stoi)).cuda().scatter_(1, labels.view(-1, 1), args.init_label).cuda()
                one_hot = F.softmax(one_hot, dim=1)
                loss = F.softmax(output, dim=1)*(F.log_softmax(output, dim=1)-torch.log(one_hot)) - torch.mul(F.softmax(output, dim=1), F.log_softmax(output, dim=1))
            if args.hard:
                loss = torch.mean(loss[selection])
            else:
                loss = torch.mean(theta_batch.detach().view(-1,1)*loss)
        acc = loss_utils.accuracy(output, labels)
        losses.update(loss.item(), labels.size(0))
        top1.update(acc, labels.size(0))
        grad_weights_on_full_train_batch = torch.autograd.grad(loss, model.parameters())
        if batch_idx > 0:
            grad_weights_on_full_train = [wb+w for wb, w in zip(grad_weights_on_full_train_batch, grad_weights_on_full_train)]
        else:
            grad_weights_on_full_train = grad_weights_on_full_train_batch
    if args.mean_grad:
        grad_weights_on_full_train = [g/len(val_iter) for g in grad_weights_on_full_train]
    return grad_weights_on_full_train,  top1.avg, losses.avg


def repass_backward(model, model_checkpoints, opt_checkpoints, outer_grads_w, train_iter, theta_mapped, theta):
    # accumulate gradients backwards to leverage hessian-vector product
    theta_grads = [torch.zeros_like(theta)]
    old_params = model_checkpoints[0]
    old_opt = opt_checkpoints[0]
    model_copy = copy.deepcopy(model)
    model_copy.load_state_dict(old_params)
    theta_sum = theta_mapped.detach().sum()
    with torch.backends.cudnn.flags(enabled=False):
        for batch_idx, batch in enumerate(train_iter):
            (inputs, lens), labels = batch.text, batch.label
            labels = batch.label
            old_params_, w_mapped = pseudo_updated_params(model_copy, old_params, old_opt, inputs, lens, labels, theta_mapped[batch.idx], theta_sum)
            grad_batch = torch.autograd.grad(w_mapped, theta, grad_outputs=outer_grads_w, retain_graph=True)
            theta_grads = [g + b for g, b in zip(theta_grads, grad_batch)]
    return theta_grads[0]

def pseudo_updated_params(pseudo_net, model_checkpoint, opt_checkpoint, inputs, lens, labels, theta, theta_sum):
    pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=args.inner_lr)
    w_old = [p for p in pseudo_net.parameters()]
    output = pseudo_net(inputs, lens)
    pseudo_loss_vector = F.cross_entropy(output, labels, reduction='none').flatten()
    pseudo_loss_vector *= theta
    pseudo_loss = torch.sum(pseudo_loss_vector/theta_sum)
    pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)
    w_mapped = pseudo_optimizer.meta_step_adam(pseudo_grads, lr=opt_checkpoint['param_groups'][0]['lr'])
    return w_old, w_mapped

def solve(model, train_loader, train_loader_backward, valid_loader, test_loader):
    if args.use_sigmoid:
        theta = torch.full([len(train_loader.dataset)], 0, dtype=torch.float, requires_grad=True, device="cuda")
    else:
        theta = torch.full([len(train_loader.dataset)], args.init_theta, dtype=torch.float, requires_grad=True, device="cuda")
    if args.optim =='Adam':
        theta_opt = Adam([theta], lr=args.outer_lr)
    elif args.optim =='SGD':
        theta_opt = SGD([theta], lr=args.outer_lr, momentum=0.9)
    theta.grad = torch.zeros_like(theta)
    best_theta = theta
    for outer_iter in range(args.max_outer_iter):
        if args.temp_anneal:
            temp = args.end_temp + (args.max_outer_iter - outer_iter)/args.max_outer_iter * (1-args.end_temp)
        else:
            temp = 1
        if args.use_sigmoid:
            theta_mapped = F.sigmoid(theta/temp)
        else:
            theta_mapped = theta
        print(theta_mapped)
        if not args.disable_outer_scheduler:
            assign_learning_rate(theta_opt, 0.5 * (1 + np.cos(np.pi * outer_iter / args.max_outer_iter)) * args.outer_lr)
        diverged = True
        while diverged:
            model_copy_converged, loss, train_acc, model_weights_cache, opt_checkpoints_cache, diverged = train_to_converge(model, train_loader, theta_mapped.detach(),args.epoch_converge,args.inner_obj)
            print(f"diverged {diverged} loss {loss}")
            print(f'train acc{train_acc}')
            if outer_iter % args.check_ft_every==0:
                model_copy_converged_ft, loss_ft, train_acc_ft, _, _, _ = train_to_converge(model, train_loader, theta_mapped.detach(), args.epoch_converge_fully_train,args.inner_obj)

        if args.stochastic_outer and args.subset_outer:
            if args.use_dev_outer:
                valid_loader = construct_outer_subloader(dev_data_all)
            else:
                valid_loader = construct_outer_subloader(train_data)
        grad_weights_on_full_train, top1_outer, loss_outer = get_grad_weights_on_valid(model_copy_converged, valid_loader, theta_mapped.detach())
        print(f"outer acc {top1_outer}, loss_outer {loss_outer}")
        grad_theta = repass_backward(model, model_weights_cache, opt_checkpoints_cache, grad_weights_on_full_train, train_loader_backward, theta_mapped, theta)
        theta_opt.zero_grad()
        print(f"sum grads {sum([g for g in grad_theta])}")
        with torch.no_grad():
            theta.grad += grad_theta.data
        torch.nn.utils.clip_grad_norm_(theta, args.clip_constant)
        theta_opt.step()
        if not args.use_sigmoid:
            with torch.no_grad():
                theta.data.clamp_(min=0, max=args.theta_upper_lim)
        torch.cuda.empty_cache()
        if outer_iter % args.check_ft_every == 0:
            test_acc1_ft, test_loss_ft = eval(model_copy_converged_ft, test_loader, name="test")
            if args.wandb:
                wandb.log({"train_loss_ft": loss_ft,"train_acc_ft":train_acc_ft,"test_acc_ft": test_acc1_ft, "test_loss_ft":test_loss_ft})
        theta_score=copy.deepcopy(theta)
        if args.wandb:
            wandb.log({"train_loss": loss, "loss_outer": loss_outer, "temp":temp})
        best_theta=theta_score

    print("++++++++++++++++finished solving++++++++++++++++++++")
    return best_theta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='summary generator')
    parser.add_argument('--seed', type=int, default=12345, metavar='seed', help='random seed (default: 0)')
    parser.add_argument('--method', type=str, default="probability_1step")
    parser.add_argument('--limit', default=1000, type=int)
    parser.add_argument('--K', default=1, type=int)
    parser.add_argument('--train_batch_size', default=32, type=int)
    parser.add_argument('--backward_batch_size', default=64, type=int)
    parser.add_argument('--outer_lr', default=5e-2, type=float)
    parser.add_argument('--inner_lr', default=1e-4, type=float)
    parser.add_argument('--div_tol', default=9, type=float)
    parser.add_argument('--outer_ratio', default=0.1, type=float)
    parser.add_argument('--theta_upper_lim', default=100, type=float)
    parser.add_argument('--outer_threshold', default=0, type=float)
    parser.add_argument('--max_outer_iter', default=250, type=int)
    parser.add_argument('--num_worker', default=4, type=int)
    parser.add_argument('--sample_times', default=1, type=int)
    parser.add_argument('--inter_dim', default=32, type=int)
    parser.add_argument('--runs_name', default="ours", type=str)
    parser.add_argument('--scheduler', default="cosine", type=str)
    parser.add_argument('--gold_data_path', default=None, type=str)
    parser.add_argument('--syn_data_path', default=None, type=str)
    parser.add_argument('--outer_obj', default="combined", type=str)
    parser.add_argument('--inner_obj', default="ce", type=str)
    parser.add_argument('--save_path', default="", type=str)
    parser.add_argument('--task_name', default="rte", type=str)
    parser.add_argument('--num_use_samples_inner', default=1000, type=int)
    parser.add_argument('--num_use_samples_outer', default=1000, type=int)
    parser.add_argument('--init_label', default=10, type=int)
    parser.add_argument('--init_theta', default=1, type=float)
    parser.add_argument('--epoch_converge', default=20, type=int)
    parser.add_argument('--epoch_converge_fully_train', default=5, type=int)
    parser.add_argument('--check_ft_every', default=10, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
    parser.add_argument("--iterative", default=False, action="store_true")
    parser.add_argument("--mean_grad", default=False, action="store_true")
    parser.add_argument("--use_test", default=False, action="store_true")
    parser.add_argument("--shuffle_train", default=False, action="store_true")
    parser.add_argument("--use_sigmoid", default=False, action="store_true")
    parser.add_argument("--hard", default=False, action="store_true")
    parser.add_argument("--use_dev_outer", default=False, action="store_true")
    parser.add_argument('--clip_constant', default=3, type=float)
    parser.add_argument('--end_temp', default=0.1, type=float)
    parser.add_argument('--wandb', action="store_true")
    parser.add_argument('--optim', type=str, default="Adam")
    parser.add_argument('--subset_outer', action="store_true")
    parser.add_argument('--stochastic_outer', action="store_true")
    parser.add_argument('--disable_outer_scheduler', action="store_true")
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--temp_anneal', action="store_true")

    args = parser.parse_args()
    randnum = random.random()


    if args.wandb:
        os.system("wandb login --relogin xxxxxx")
        wandb.init(project=f"your-project-name")

    set_seed(args.seed)

    print(f"learning rate: {args.inner_lr}")
    print(f"seed: {args.seed}")

    print('num of use syn samples:{}'.format(args.num_use_samples_inner))
    train_iter, train_iter_backward, dev_iter, test_iter, TEXT, LABEL, train_data, dev_data_all = load_iters(args.train_batch_size, args.backward_batch_size, device, args.gold_data_path, args.syn_data_path, vectors, False, args.num_use_samples_inner, args.num_use_samples_outer,args.shuffle_train)

    tmp=len(LABEL.vocab.stoi)
    print(f'num of lable{tmp}')

    model = RNN(len(TEXT.vocab), len(LABEL.vocab.stoi),
                 EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT_RATE, LAYER_NUM,
                 TEXT.vocab.vectors, freeze).to(device)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    loss_func = nn.CrossEntropyLoss()
    if args.use_test:
        dev_iter = test_iter
    args.save_path=os.path.join(args.syn_data_path,f'best_thetas_inner{args.num_use_samples_inner}_outter{args.num_use_samples_outer}')
    args.save_path = f"{args.save_path}/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f'use {len(train_iter.dataset.examples)} train data...')
    print(f'use {len(dev_iter.dataset.examples)} dev data...')
    print(f'use {len(test_iter.dataset.examples)} test data...')
    best_theta = solve(model, train_iter, train_iter_backward, dev_iter, test_iter)
    torch.save(best_theta, f"{args.save_path}/best_thetas.pth")
    print(f"best thetas saved to {args.save_path}/best_thetas.pth")
    print(f"best thetas {best_theta}")