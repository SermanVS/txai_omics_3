import torch
from src.models.tabular.base import BaseModel
from .repository.models import STGClassificationModel, STGRegressionModel
import pandas as pd


class StochasticGatesModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        if self.hparams.task == "classification":
            self.model = STGClassificationModel(
                input_dim=self.hparams.input_dim,
                nr_classes=self.hparams.output_dim,
                hidden_dims=self.hparams.hidden_dims,
                activation=self.hparams.activation,
                sigma=self.hparams.sigma,
                lam=self.hparams.lam
            )
        elif self.hparams.task == "regression":
            self.model = STGRegressionModel(
                input_dim=self.hparams.input_dim,
                output_dim=self.hparams.output_dim,
                hidden_dims=self.hparams.hidden_dims,
                activation=self.hparams.activation,
                sigma=self.hparams.sigma,
                lam=self.hparams.lam
            )

    def forward(self, batch):
        if isinstance(batch, dict):
            x = {'input': batch["all"], 'label': batch["target"]}
        else:
            x =  {'input': batch}
        x = self.model(x)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x

    def calc_out_and_loss(self, out, y, stage):
        if stage == "trn":
            loss = self.loss_fn(out, y)
            reg = torch.mean(self.model.reg((self.model.mu + 0.5) / self.model.sigma))
            loss = loss + self.model.lam * reg
            return out, loss
        elif stage == "val":
            loss = self.loss_fn(out, y)
            return out, loss
        elif stage == "tst":
            loss = self.loss_fn(out, y)
            return out, loss
        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def get_feature_importance(self, data, feature_names, method="shap_kernel"):
        if method == "native":
            importance_values = self.model.get_gates(mode="prob")
            feature_importances = pd.DataFrame.from_dict(
                {
                    'feature': feature_names['all'],
                    'importance': importance_values
                }
            )
            return feature_importances
        else:
            return super().get_feature_importance(data, feature_names, method)
