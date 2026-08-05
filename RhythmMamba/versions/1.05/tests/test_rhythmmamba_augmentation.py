import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from neural_methods.trainer.RhythmMambaTrainer import RhythmMambaTrainer


def make_trainer():
    trainer = RhythmMambaTrainer.__new__(RhythmMambaTrainer)
    trainer.data_dict = {}
    trainer.diff_flag = 0
    trainer.config = SimpleNamespace(
        VALID=SimpleNamespace(DATA=SimpleNamespace(FS=30))
    )
    return trainer


class RhythmMambaAugmentationTest(unittest.TestCase):
    def test_high_hr_upsampling_only_changes_current_sample(self):
        trainer = make_trainer()
        data = torch.arange(12, dtype=torch.float32).view(2, 6, 1, 1, 1)
        labels = torch.arange(12, dtype=torch.float32).view(2, 6)

        with patch(
            "neural_methods.trainer.RhythmMambaTrainer.np.random.random",
            side_effect=[np.array([0.0, 1.0]), np.array([1.0, 1.0])],
        ), patch(
            "neural_methods.trainer.RhythmMambaTrainer.random.randint",
            return_value=0,
        ), patch(
            "neural_methods.trainer.RhythmMambaTrainer.calculate_hr",
            return_value=(100.0, 100.0),
        ):
            data_aug, labels_aug = trainer.data_augmentation(
                data, labels, ["subject1", "subject2"], ["0", "0"]
            )

        expected_first = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        self.assertTrue(torch.allclose(data_aug[0, :, 0, 0, 0], expected_first))
        self.assertTrue(torch.allclose(labels_aug[0], expected_first))
        self.assertTrue(torch.equal(data_aug[1], data[1]))
        self.assertTrue(torch.equal(labels_aug[1], labels[1]))

    def test_low_hr_downsampling_only_changes_current_sample(self):
        trainer = make_trainer()
        data = torch.arange(12, dtype=torch.float32).view(2, 6, 1, 1, 1)
        labels = torch.arange(12, dtype=torch.float32).view(2, 6)

        with patch(
            "neural_methods.trainer.RhythmMambaTrainer.np.random.random",
            side_effect=[np.array([0.0, 1.0]), np.array([1.0, 1.0])],
        ), patch(
            "neural_methods.trainer.RhythmMambaTrainer.calculate_hr",
            return_value=(60.0, 60.0),
        ):
            data_aug, labels_aug = trainer.data_augmentation(
                data, labels, ["subject1", "subject2"], ["0", "0"]
            )

        expected_first = torch.tensor([0.0, 2.0, 4.0, 0.0, 2.0, 4.0])
        self.assertTrue(torch.equal(data_aug[0, :, 0, 0, 0], expected_first))
        self.assertTrue(torch.equal(labels_aug[0], expected_first))
        self.assertTrue(torch.equal(data_aug[1], data[1]))
        self.assertTrue(torch.equal(labels_aug[1], labels[1]))

    def test_horizontal_flip_is_per_sample(self):
        trainer = make_trainer()
        data = torch.tensor(
            [
                [[[[1.0, 2.0, 3.0]]]],
                [[[[4.0, 5.0, 6.0]]]],
            ]
        )
        labels = torch.tensor([[1.0], [2.0]])

        with patch(
            "neural_methods.trainer.RhythmMambaTrainer.np.random.random",
            side_effect=[np.array([1.0, 1.0]), np.array([0.0, 1.0])],
        ):
            data_aug, labels_aug = trainer.data_augmentation(
                data, labels, ["subject1", "subject2"], ["0", "0"]
            )

        self.assertTrue(torch.equal(data_aug[0, 0, 0, 0], torch.tensor([3.0, 2.0, 1.0])))
        self.assertTrue(torch.equal(data_aug[1], data[1]))
        self.assertTrue(torch.equal(labels_aug, labels))


if __name__ == "__main__":
    unittest.main()
