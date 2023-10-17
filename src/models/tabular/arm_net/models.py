import torch
from src.models.tabular.base import BaseModel
from .repository.models.lr import LRModel
from .repository.models.fm import FMModel
from .repository.models.hofm import HOFMModel
from .repository.models.afm import AFMModel
from .repository.models.dcn import CrossNetModel
from .repository.models.xdfm import CINModel
from .repository.models.dnn import DNNModel
from .repository.models.gcn import GCNModel
from .repository.models.gat import GATModel
from .repository.models.wd import WDModel
from .repository.models.pnn import IPNNModel
from .repository.models.pnn import KPNNModel
from .repository.models.nfm import NFMModel
from .repository.models.dfm import DeepFMModel
from .repository.models.dcn import DCNModel
from .repository.models.xdfm import xDeepFMModel
from .repository.models.afn import AFNModel
from .repository.models.armnet import ARMNetModel
from .repository.models.armnet_1h import ARMNetModel as ARMNet1H
from .repository.models.gc_arm import GC_ARMModel
from .repository.models.sa_glu import SA_GLUModel
import numpy as np


class ARMNetModels(BaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build_network()

    def build_network(self):
        if self.hparams.model == 'lr':
            self.model = LRModel(self.hparams.nfeat)
        elif self.hparams.model == 'fm':
            self.model = FMModel(self.hparams.nfeat, self.hparams.nemb)
        elif self.hparams.model == 'hofm':
            self.model = HOFMModel(self.hparams.nfeat, self.hparams.nemb, self.hparams.k)
        elif self.hparams.model == 'afm':
            self.model = AFMModel(self.hparams.nfeat, self.hparams.nemb, self.hparams.h, self.hparams.dropout)
        elif self.hparams.model == 'dcn':
            self.model = CrossNetModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                cn_layers=self.hparams.k,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'cin':
            self.model = CINModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                cin_layers=self.hparams.k,
                nfilter=self.hparams.h,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'afn':
            self.model = AFNModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                afn_hid=self.hparams.h,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_layers=self.hparams.dnn_nlayer,
                deep_hid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'armnet':
            self.model = ARMNetModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                nhead=self.hparams.nattn_head,
                alpha=self.hparams.alpha,
                nhid=self.hparams.h,
                mlp_nlayer=self.hparams.mlp_nlayer,
                mlp_nhid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_nlayer=self.hparams.dnn_nlayer,
                deep_nhid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'armnet_1h':
            self.model = ARMNet1H(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                alpha=self.hparams.alpha,
                nhid=self.hparams.h,
                d_k=self.hparams.nemb,
                mlp_nlayer=self.hparams.mlp_nlayer,
                mlp_nhid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_nlayer=self.hparams.dnn_nlayer,
                deep_nhid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'dnn':
            self.model = DNNModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'gcn':
            self.model = GCNModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                gcn_layers=self.hparams.k,
                gcn_hid=self.hparams.h,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'gat':
            self.model = GATModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                gat_layers=self.hparams.k,
                gat_hid=self.hparams.h,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                alpha=0.2,
                nhead=self.hparams.nattn_head,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'wd':
            self.model = WDModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'ipnn':
            self.model = IPNNModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'kpnn':
            self.model = KPNNModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'nfm':
            self.model = NFMModel(
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'dfm':
            self.model = DeepFMModel(self.hparams.nfield, self.hparams.nfeat, self.hparams.nemb, self.hparams.mlp_nlayer, self.hparams.mlp_nhid, self.hparams.dropout)
        elif self.hparams.model == 'dcn_plus':
            self.model = DCNModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                cn_layers=self.hparams.k,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'xdfm':
            self.model = xDeepFMModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                cin_layers=self.hparams.k,
                nfilter=self.hparams.h,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'gc_arm':
            self.model = GC_ARMModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                nhead=self.hparams.nattn_head,
                alpha=self.hparams.alpha,
                arm_hid=self.hparams.h,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_layers=self.hparams.dnn_nlayer,
                deep_hid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        elif self.hparams.model == 'sa_glu':
            self.model = SA_GLUModel(
                nfield=self.hparams.nfield,
                nfeat=self.hparams.nfeat,
                nemb=self.hparams.nemb,
                mlp_layers=self.hparams.mlp_nlayer,
                mlp_hid=self.hparams.mlp_nhid,
                dropout=self.hparams.dropout,
                ensemble=self.hparams.ensemble,
                deep_layers=self.hparams.dnn_nlayer,
                deep_hid=self.hparams.dnn_nhid,
                noutput=self.hparams.output_dim
            )
        else:
            raise ValueError(f'Unsupported model: {self.hparams.model}')

    def forward(self, batch):
        if isinstance(batch, dict):
            batch_size = batch["all"].shape[0]
            feats_ids = torch.LongTensor(np.arange(self.hparams.nfeat)).to(batch['all'].device)
            feats_ids = feats_ids.repeat(batch["all"].shape[0], 1)
            x = {
                'id': feats_ids,
                'value': batch["all"],
            }
        else:
            batch_size = batch.shape[0]
            feats_ids = torch.LongTensor(np.arange(self.hparams.nfeat)).to(batch.device)
            feats_ids = feats_ids.repeat(batch.shape[0], 1)
            x = {
                'id': feats_ids,
                'value': batch,
            }
        x = self.model(x)
        x = x.view(batch_size, -1)
        if self.produce_probabilities:
            return torch.softmax(x, dim=1)
        else:
            return x
