import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 
import six
import collections

SKIP_TYPES = six.string_types


class Identity(nn.Module):
    def forward(self, *args):
        if len(args) == 1:
            return args[0]
        return args

def get_batcnnorm(bn, nr_features=None, nr_dims=1):
    if isinstance(bn, nn.Module):
        return bn

    assert 1 <= nr_dims <= 3

    if bn in (True, 'async'):
        clz_name = 'BatchNorm{}d'.format(nr_dims)
        return getattr(nn, clz_name)(nr_features)
    else:
        raise ValueError('Unknown type of batch normalization: {}.'.format(bn))


def get_dropout(dropout, nr_dims=1):
    if isinstance(dropout, nn.Module):
        return dropout

    if dropout is True:
        dropout = 0.5
    if nr_dims == 1:
        return nn.Dropout(dropout, True)
    else:
        clz_name = 'Dropout{}d'.format(nr_dims)
        return getattr(nn, clz_name)(dropout)


def get_activation(act):
    if isinstance(act, nn.Module):
        return act

    assert type(act) is str, 'Unknown type of activation: {}.'.format(act)
    act_lower = act.lower()
    if act_lower == 'identity':
        return Identity()
    elif act_lower == 'relu':
        return nn.ReLU(True)
    elif act_lower == 'selu':
        return nn.SELU(True)
    elif act_lower == 'sigmoid':
        return nn.Sigmoid()
    elif act_lower == 'tanh':
        return nn.Tanh()
    else:
        try:
            return getattr(nn, act)
        except AttributeError:
            raise ValueError('Unknown activation function: {}.'.format(act))


def get_optimizer(optimizer, model, *args, **kwargs):
    if isinstance(optimizer, (optim.Optimizer)):
        return optimizer

    if type(optimizer) is str:
        try:
            optimizer = getattr(optim, optimizer)
        except AttributeError:
            raise ValueError('Unknown optimizer type: {}.'.format(optimizer))
    return optimizer(filter(lambda p: p.requires_grad, model.parameters()), *args, **kwargs)
    

def stmap(func, iterable):
    if isinstance(iterable, six.string_types):
        return func(iterable)
    elif isinstance(iterable, (collections.Sequence, collections.UserList)):
        return [stmap(func, v) for v in iterable]
    elif isinstance(iterable, collections.Set):
        return {stmap(func, v) for v in iterable}
    elif isinstance(iterable, (collections.Mapping, collections.UserDict)):
        return {k: stmap(func, v) for k, v in iterable.items()}
    else:
        return func(iterable)


def _as_tensor(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        return o
    if torch.is_tensor(o):
        return o
    return torch.from_numpy(np.array(o))


def as_tensor(obj):
    return stmap(_as_tensor, obj)


def _as_numpy(o):
    from torch.autograd import Variable
    if isinstance(o, SKIP_TYPES):
        return o
    if isinstance(o, Variable):
        o = o
    if torch.is_tensor(o):
        return o.cpu().numpy()
    return np.array(o)


def as_numpy(obj):
    return stmap(_as_numpy, obj)


def _as_float(o):
    if isinstance(o, SKIP_TYPES):
        return o
    if torch.is_tensor(o):
        return o.item()
    arr = as_numpy(o)
    assert arr.size == 1
    return float(arr)


def as_float(obj):
    return stmap(_as_float, obj)


def _as_cpu(o):
    from torch.autograd import Variable
    if isinstance(o, Variable) or torch.is_tensor(o):
        return o.cpu()
    return o


def as_cpu(obj):
    return stmap(_as_cpu, obj)


## For synthetic dataset creation
import math
from sklearn.datasets import make_moons
from scipy.stats import norm


# Create a simple dataset
def create_twomoon_dataset(n, p):
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=0.1, random_state=None)
    print(y.shape)
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    print(data.shape)
    return data, y


def create_sin_dataset(n,p):
    x1=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    x2=5*(np.random.uniform(0,1,n)).reshape(-1,1)
    y=np.sin(x1)*np.cos(x2)**3
    relevant=np.hstack((x1,x2))
    noise_vector = norm.rvs(loc=0, scale=1, size=[n,p-2])
    data = np.concatenate([relevant, noise_vector], axis=1)
    return data, y.astype(np.float32)



def create_simple_sin_dataset(n, p):
    '''This dataset was added to provide an example of L1 norm reg failure for presentation.
    '''
    assert p == 2
    x1 = np.random.uniform(-math.pi, math.pi, n).reshape(n ,1)
    x2 = np.random.uniform(-math.pi, math.pi, n).reshape(n, 1)
    y = np.sin(x1)
    data = np.concatenate([x1, x2], axis=1)
    print("data.shape: {}".format(data.shape))
    return data, y
    
