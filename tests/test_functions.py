from txai_omics_3.models.tabular.widedeep.ft_transformer import WDFTTransformerModel
import torch
import unittest

class TestModelInitFunction(unittest.TestCase):
    def test_query(self):
        # Model
        fn_model = f"data/immuno/model.ckpt"
        model = WDFTTransformerModel.load_from_checkpoint(fn_model)
        return True

if __name__ == '__main__':
    unittest.main()