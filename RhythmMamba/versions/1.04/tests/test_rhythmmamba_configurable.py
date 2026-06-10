import unittest

import torch

from neural_methods.model.RhythmMamba import RhythmMamba


class RhythmMambaConfigurableTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Mamba forward requires CUDA in this environment.")
    def test_custom_mamba_parameters_keep_output_shape(self):
        model = RhythmMamba(
            depth=2,
            embed_dim=48,
            mlp_ratio=2,
            drop_path_rate=0.05,
            mamba_d_state=16,
            mamba_d_conv=3,
            mamba_expand=1,
            multi_temporal_paths=2,
        ).cuda()
        model.eval()
        data = torch.randn(1, 32, 3, 64, 64, device="cuda")

        with torch.no_grad():
            output = model(data)

        self.assertEqual(output.shape, (1, 32))

    def test_invalid_temporal_path_raises_clear_error(self):
        model = RhythmMamba(
            depth=1,
            embed_dim=48,
            mamba_d_state=16,
            mamba_d_conv=3,
            mamba_expand=1,
            multi_temporal_paths=4,
        )
        model.eval()
        data = torch.randn(1, 20, 3, 64, 64)

        with self.assertRaisesRegex(ValueError, "Temporal length"):
            with torch.no_grad():
                model(data)


if __name__ == "__main__":
    unittest.main()
