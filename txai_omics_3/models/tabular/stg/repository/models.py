import torch.nn as nn
import torch
import numpy as np
from .layers import MLPLayer, FeatureSelector


__all__ = ['MLPModel', 'STGRegressionModel', 'STGClassificationModel']


class ModelIOKeysMixin(object):
    def _get_input(self, feed_dict):
        return feed_dict['input']

    def _get_label(self, feed_dict):
        return feed_dict['label']

    def _get_covariate(self, feed_dict):
        '''For cox'''
        return feed_dict['X']

    def _get_fail_indicator(self, feed_dict):
        '''For cox'''
        return feed_dict['E'].reshape(-1, 1)

    def _get_failure_time(self, feed_dict):
        '''For cox'''
        return feed_dict['T']

    def _compose_output(self, value):
        return dict(pred=value)


class MLPModel(MLPLayer):
    def freeze_weights(self):
        for name, p in self.named_parameters():
            if name != 'mu':
                p.requires_grad = False

    def get_gates(self, mode):
        if mode == 'raw':
            return self.mu.detach().cpu().numpy()
        elif mode == 'prob':
            return np.minimum(1.0, np.maximum(0.0, self.mu.detach().cpu().numpy() + 0.5)) 
        else:
            raise NotImplementedError()


class STGRegressionModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, output_dim, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma)
        self.loss = nn.MSELoss()
        self.reg = self.FeatureSelector.regularizer 
        self.lam = lam
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma

    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        pred = super().forward(x)
        return pred
    

class STGClassificationModel(MLPModel, ModelIOKeysMixin):
    def __init__(self, input_dim, nr_classes, hidden_dims, batch_norm=None, dropout=None, activation='relu',
                 sigma=1.0, lam=0.1):
        super().__init__(input_dim, nr_classes, hidden_dims,
                         batch_norm=batch_norm, dropout=dropout, activation=activation)
        self.FeatureSelector = FeatureSelector(input_dim, sigma)
        self.softmax = nn.Softmax()
        self.loss = nn.CrossEntropyLoss()
        self.reg = self.FeatureSelector.regularizer
        self.lam = lam 
        self.mu = self.FeatureSelector.mu
        self.sigma = self.FeatureSelector.sigma
        
    def forward(self, feed_dict):
        x = self.FeatureSelector(self._get_input(feed_dict))
        logits = super().forward(x)
        return logits

    def _compose_output(self, logits):
        value = self.softmax(logits)
        _, pred = value.max(dim=1)
        return dict(prob=value, pred=pred, logits=logits)
