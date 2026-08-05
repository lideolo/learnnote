import unittest
from types import SimpleNamespace

import torch

from neural_methods.trainer.RhythmMambaTrainer import RhythmMambaTrainer


class FirstPixelModel:
    def __call__(self, data):
        return data[:, :, 0, 0, 0]


def make_trainer(horizontal_flip=False, align_sign=True):
    trainer = RhythmMambaTrainer.__new__(RhythmMambaTrainer)
    trainer.model = FirstPixelModel()
    trainer.config = SimpleNamespace(
        INFERENCE=SimpleNamespace(
            TTA=SimpleNamespace(
                HORIZONTAL_FLIP=horizontal_flip,
                ALIGN_SIGN=align_sign,
            )
        )
    )
    return trainer


def normalize(signal):
    return (signal - torch.mean(signal, axis=-1).view(-1, 1)) / torch.std(signal, axis=-1).view(-1, 1)


class RhythmMambaTTATest(unittest.TestCase):
    def test_predict_without_tta_matches_single_forward(self):
        trainer = make_trainer(horizontal_flip=False)
        data = torch.tensor([[[[[1.0, 5.0]]], [[[2.0, 6.0]]], [[[4.0, 8.0]]]]])

        pred = trainer._predict_ppg(data)

        self.assertTrue(torch.allclose(pred, normalize(torch.tensor([[1.0, 2.0, 4.0]]))))

    def test_horizontal_flip_tta_averages_original_and_flipped_predictions(self):
        trainer = make_trainer(horizontal_flip=True, align_sign=False)
        data = torch.tensor([[[[[1.0, 5.0]]], [[[2.0, 6.0]]], [[[4.0, 8.0]]]]])

        pred = trainer._predict_ppg(data)

        self.assertTrue(torch.allclose(pred, normalize(torch.tensor([[3.0, 4.0, 6.0]]))))


if __name__ == "__main__":
    unittest.main()
