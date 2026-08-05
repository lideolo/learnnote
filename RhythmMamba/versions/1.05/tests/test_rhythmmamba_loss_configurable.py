import unittest
from unittest.mock import patch

import torch

from neural_methods.loss.TorchLossComputer import Hybrid_Loss, TorchLossComputer


class ConstantLoss(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, *args, **kwargs):
        return torch.tensor(self.value)


class RhythmMambaLossConfigurableTest(unittest.TestCase):
    def test_default_weights_match_original_hybrid_loss_formula(self):
        criterion = Hybrid_Loss()
        criterion.criterion_Pearson = ConstantLoss(4.0)

        with patch.object(
            TorchLossComputer,
            "Frequency_loss",
            return_value=(torch.tensor(2.0), torch.tensor(3.0)),
        ):
            loss = criterion(torch.ones(8), torch.ones(8), 0, 30, False)

        self.assertAlmostEqual(loss.item(), 2.8)

    def test_custom_weights_include_frequency_kl_term(self):
        criterion = Hybrid_Loss(
            time_weight=0.1,
            freq_ce_weight=0.7,
            freq_kl_weight=0.2,
            freq_std=2.0,
        )
        criterion.criterion_Pearson = ConstantLoss(4.0)

        calls = {}

        def fake_frequency_loss(*args, **kwargs):
            calls["std"] = kwargs["std"]
            return torch.tensor(2.0), torch.tensor(3.0)

        with patch.object(TorchLossComputer, "Frequency_loss", side_effect=fake_frequency_loss):
            loss = criterion(torch.ones(8), torch.ones(8), 0, 30, False)

        self.assertEqual(calls["std"], 2.0)
        self.assertAlmostEqual(loss.item(), 2.4, places=6)


if __name__ == "__main__":
    unittest.main()
