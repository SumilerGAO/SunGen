import torch
import torch.nn.functional as F
import logging
import numpy as np

def cross_entropy(K, alpha, y, weights, lmbda):
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    loss_value = torch.mean(loss(torch.matmul(K, alpha), y.long()) * weights)
    if lmbda > 0:
        loss_value += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss_value

def cross_entropy_normal(model, X, y, weights=None, reduce=True):
    # loss = torch.nn.CrossEntropyLoss(reduction='none')
    output = model(X)
    loss = F.cross_entropy(output, y.long(), reduction='none')
    if not reduce:
        return loss, F.softmax(output).detach()
    if weights is None:
        weights = torch.ones_like(loss)
    loss_value = torch.mean(loss * weights)
    return loss_value, output

def cross_entropy_proxy(model, X, y, proxy=None, reduce=True):
    # loss = torch.nn.CrossEntropyLoss(reduction='none')
    output, features = model(X, return_feat=True)
    loss = F.cross_entropy(output, y.long(), reduction='none')
    if not reduce:
        return loss, F.softmax(output).detach()
    if proxy is None:
        weights = torch.ones_like(loss)
    else:
        weights = proxy(features.detach())
        logging.info(f"scores {weights.detach().cpu().tolist()}")
    loss_reshaped=torch.reshape(loss, (-1, 1))
    loss_value = torch.mean(loss_reshaped * weights)
    return loss_value, output

def accuracy(output, labels):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # print(output)
        predicts = output.argmax(-1).reshape(-1)
        correct_num = (predicts == labels).sum().item()
        err_num = (predicts != labels).sum().item()
        acc = correct_num / (correct_num + err_num)
        return acc

def weighted_mse(K, alpha, y, weights, lmbda):
    loss = torch.mean(torch.sum((torch.matmul(K, alpha) - y) ** 2, dim=1) * weights)
    if lmbda > 0:
        loss += lmbda * torch.trace(torch.matmul(alpha.T, torch.matmul(K, alpha)))
    return loss

def cal_perclass_inds(y, inds):
    with torch.no_grad():
        per_class_inds = [sum([y[ind] == i for ind in inds]).item() for i in range(10)]
        return per_class_inds

def class_balanced_sampling(dataset, start_size):
    samples_perclass = start_size//10
    per_class_inds = [np.arange(0,len(dataset))[dataset.targets==i] for i in range(10)]
    start_subset = [np.random.choice(inds, samples_perclass) for inds in per_class_inds]
    return np.concatenate(start_subset)
# def accuracy(K, alpha, y):
#     nr_correct = torch.sum(torch.argmax(torch.matmul(K, alpha), dim=1) == y.long())
#     return 1. * nr_correct / K.shape[0]
