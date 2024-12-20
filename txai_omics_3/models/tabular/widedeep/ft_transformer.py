from .base import WDBaseModel
from pytorch_widedeep.models import FTTransformer
from pathlib import Path
import importlib.resources as rc
import data.immuno

data_dir = rc.files(data.immuno)

def get_file(file):
    with rc.as_file(data_dir.joinpath(file)) as path:
        return path

FN_SHAP = get_file('shap.pickle')
FN_CHECKPOINT = get_file('model.ckpt')
TRAIN_DATA_PATH = get_file('data.xlsx')


class WDFTTransformerModel(WDBaseModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_network(self):
        self.model = FTTransformer(
            column_idx=self.hparams.column_idx,
            cat_embed_input=self.hparams.cat_embed_input,
            cat_embed_dropout=self.hparams.cat_embed_dropout,
            use_cat_bias=self.hparams.use_cat_bias,
            cat_embed_activation=self.hparams.cat_embed_activation,
            full_embed_dropout=self.hparams.full_embed_dropout,
            shared_embed=self.hparams.shared_embed,
            add_shared_embed=self.hparams.add_shared_embed,
            frac_shared_embed=self.hparams.frac_shared_embed,
            continuous_cols=self.hparams.continuous_cols,
            cont_norm_layer=self.hparams.cont_norm_layer,
            cont_embed_dropout=self.hparams.cont_embed_dropout,
            use_cont_bias=self.hparams.use_cont_bias,
            cont_embed_activation=self.hparams.cont_embed_activation,
            input_dim=self.hparams.embed_dim,
            kv_compression_factor=self.hparams.kv_compression_factor,
            kv_sharing=self.hparams.kv_sharing,
            n_heads=self.hparams.n_heads,
            use_qkv_bias=self.hparams.use_qkv_bias,
            n_blocks=self.hparams.n_blocks,
            attn_dropout=self.hparams.attn_dropout,
            ff_dropout=self.hparams.ff_dropout,
            transformer_activation=self.hparams.transformer_activation,
            ff_factor=self.hparams.ff_factor,
            mlp_hidden_dims=self.hparams.mlp_hidden_dims,
            mlp_activation=self.hparams.mlp_activation,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_batchnorm=self.hparams.mlp_batchnorm,
            mlp_batchnorm_last=self.hparams.mlp_batchnorm_last,
            mlp_linear_first=self.hparams.mlp_linear_first,
        )
