import unittest

import torch

from neural_methods.loss.TorchLossComputer import Hybrid_Loss
from neural_methods.model.RhythmMamba import RhythmMamba


class RhythmMambaV110Test(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Mamba forward requires CUDA in this environment.")
    def test_v110_full_branch_returns_aux_and_backpropagates(self):
        model = RhythmMamba(
            depth=2,
            embed_dim=48,
            mamba_d_state=16,
            mamba_d_conv=3,
            mamba_expand=1,
            multi_temporal_paths=2,
            use_roi_stem=True,
            roi_count=5,
            use_spectral_gate=True,
            use_periodic_pe=True,
            use_periodic_modulation=True,
            return_aux=True,
            sampling_rate=30.0,
        ).cuda()
        criterion = Hybrid_Loss(
            time_weight=0.5,
            roi_phase_weight=0.03,
            harmonic_weight=0.02,
            aux_hr_weight=0.05,
        )
        data = torch.randn(2, 32, 3, 64, 64, device="cuda")
        labels = torch.randn(2, 32, device="cuda")

        pred_ppg, aux = model(data)
        self.assertEqual(pred_ppg.shape, (2, 32))
        self.assertEqual(aux["roi_tokens"].shape, (2, 16, 5, 48))
        self.assertEqual(aux["roi_weights"].shape, (2, 5))
        self.assertEqual(aux["hr_logits"].shape, (2, 106))

        loss = 0.0
        for sample_idx in range(2):
            sample_aux = {
                key: value[sample_idx] if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == 2 else value
                for key, value in aux.items()
            }
            loss = loss + criterion(pred_ppg[sample_idx], labels[sample_idx], 0, 30, False, aux=sample_aux)
        loss = loss / 2
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
