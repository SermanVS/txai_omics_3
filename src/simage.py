from src.models.tabular.widedeep.ft_transformer import WDFTTransformerModel
import pandas as pd
import torch
import warnings
warnings.filterwarnings("ignore", ".*will save all targets and predictions in the buffer. For large datasets, "
                                  "this may lead to large memory footprint.*")


def inference(
        model_fn: str,
        df: pd.DataFrame,
) -> None:
    """ Inference the SImAge model

    :param model_fn: Filename of the model.
    :param df: Dataframe containing the immunomarkers and chronological age.
    """

    feats = [
        'CXCL9',
        'CCL22',
        'IL6',
        'PDGFB',
        'CD40LG',
        'IL27',
        'VEGFA',
        'CSF1',
        'PDGFA',
        'CXCL10'
    ]

    if not set(feats + ['Age']).issubset(df.columns):
        raise ValueError(f"Not all features are present in the dataframe.")

    model = WDFTTransformerModel.load_from_checkpoint(checkpoint_path=model_fn)
    model.eval()
    model.freeze()

    df['SImAge'] = model(torch.from_numpy(df.loc[:, feats].values)).cpu().detach().numpy().ravel()
    df['SImAge acceleration'] = df['SImAge'] - df['Age']
    df['|SImAge acceleration|'] = df['SImAge acceleration'].abs()
