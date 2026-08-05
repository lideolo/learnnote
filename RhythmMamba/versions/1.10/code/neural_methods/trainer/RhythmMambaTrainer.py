"""Trainer for RhythmMamba."""
import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr, calculate_metric_per_video
from evaluation.metrics import calculate_metrics
from neural_methods.model.RhythmMamba import  RhythmMamba
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import Hybrid_Loss

def build_hybrid_loss(config):
    loss_config = config.TRAIN.LOSS
    return Hybrid_Loss(
        time_weight=loss_config.TIME_WEIGHT,
        freq_ce_weight=loss_config.FREQ_CE_WEIGHT,
        freq_kl_weight=loss_config.FREQ_KL_WEIGHT,
        freq_std=loss_config.FREQ_STD,
        roi_phase_weight=loss_config.ROI_PHASE_WEIGHT,
        harmonic_weight=loss_config.HARMONIC_WEIGHT,
        aux_hr_weight=loss_config.AUX_HR_WEIGHT,
        aux_hr_std=loss_config.AUX_HR_STD,
    )

def build_rhythmmamba_model(config):
    model_config = config.MODEL.RHYTHMMAMBA
    return RhythmMamba(
        depth=model_config.DEPTH,
        embed_dim=model_config.EMBED_DIM,
        mlp_ratio=model_config.MLP_RATIO,
        drop_rate=config.MODEL.DROP_RATE,
        drop_path_rate=model_config.DROP_PATH_RATE,
        mamba_d_state=model_config.MAMBA_D_STATE,
        mamba_d_conv=model_config.MAMBA_D_CONV,
        mamba_expand=model_config.MAMBA_EXPAND,
        multi_temporal_paths=model_config.MULTI_TEMPORAL_PATHS,
        use_roi_stem=model_config.USE_ROI_STEM,
        roi_count=model_config.ROI_COUNT,
        roi_residual_scale=model_config.ROI_RESIDUAL_SCALE,
        use_spectral_gate=model_config.USE_SPECTRAL_GATE,
        use_periodic_pe=model_config.USE_PERIODIC_PE,
        use_periodic_modulation=model_config.USE_PERIODIC_MODULATION,
        return_aux=model_config.RETURN_AUX,
        sampling_rate=config.TRAIN.DATA.FS if config.TRAIN.DATA.FS else 30.0,
        spectral_hr_low=model_config.SPECTRAL_HR_LOW,
        spectral_hr_high=model_config.SPECTRAL_HR_HIGH,
        spectral_prior_strength=model_config.SPECTRAL_PRIOR_STRENGTH,
        spectral_adaptive_strength=model_config.SPECTRAL_ADAPTIVE_STRENGTH,
    )

class RhythmMambaTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.config = config
        self.min_valid_loss = None
        self.best_valid_score = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.data_dict = {}
        self.dataset = config.TRAIN.DATA.DATASET
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = build_rhythmmamba_model(config).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            self._load_resume_checkpoint(config)
            self.num_train_batches = len(data_loader["train"])
            self.criterion = build_hybrid_loss(config)
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=0)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            self.model = build_rhythmmamba_model(config).to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")

    def _load_resume_checkpoint(self, config):
        if not config.MODEL.RESUME:
            return
        if not os.path.exists(config.MODEL.RESUME):
            raise ValueError("Resume model path error! Please check MODEL.RESUME in your yaml.")
        checkpoint = torch.load(config.MODEL.RESUME, map_location=self.device)
        load_result = self.model.load_state_dict(
            checkpoint, strict=config.MODEL.RHYTHMMAMBA.LOAD_STRICT)
        print("Loaded resume checkpoint: {}".format(config.MODEL.RESUME))
        if not config.MODEL.RHYTHMMAMBA.LOAD_STRICT:
            print("Resume missing keys: {}".format(load_result.missing_keys))
            print("Resume unexpected keys: {}".format(load_result.unexpected_keys))

    def _normalize_ppg(self, pred_ppg):
        return (pred_ppg-torch.mean(pred_ppg, axis=-1).view(-1, 1))/torch.std(pred_ppg, axis=-1).view(-1, 1)

    def _unpack_model_output(self, model_output):
        if isinstance(model_output, tuple):
            return model_output[0], model_output[1]
        return model_output, None

    def _slice_aux(self, aux, sample_index, batch_size):
        if aux is None:
            return None
        sample_aux = {}
        for key, value in aux.items():
            if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
                sample_aux[key] = value[sample_index]
            else:
                sample_aux[key] = value
        return sample_aux

    def _predict_ppg(self, data):
        pred_ppg, _ = self._unpack_model_output(self.model(data))
        tta_config = self.config.INFERENCE.TTA
        if tta_config.HORIZONTAL_FLIP:
            pred_ppg_flip, _ = self._unpack_model_output(self.model(torch.flip(data, dims=[4])))
            if tta_config.ALIGN_SIGN:
                pred_centered = pred_ppg - torch.mean(pred_ppg, axis=-1, keepdim=True)
                flip_centered = pred_ppg_flip - torch.mean(pred_ppg_flip, axis=-1, keepdim=True)
                same_sign = torch.sum(pred_centered * flip_centered, dim=-1, keepdim=True) >= 0
                pred_ppg_flip = torch.where(same_sign, pred_ppg_flip, -pred_ppg_flip)
            pred_ppg = (pred_ppg + pred_ppg_flip) / 2
        return self._normalize_ppg(pred_ppg)

    def _store_batch_predictions(self, predictions, labels, pred_ppg, label_ppg, batch, batch_size, chunk_len):
        label_ppg = label_ppg.view(-1, 1)
        pred_ppg = pred_ppg.view(-1, 1)
        for ib in range(batch_size):
            subj_index = batch[2][ib]
            sort_index = int(batch[3][ib])
            if subj_index not in predictions.keys():
                predictions[subj_index] = dict()
                labels[subj_index] = dict()
            predictions[subj_index][sort_index] = pred_ppg[ib * chunk_len:(ib + 1) * chunk_len]
            labels[subj_index][sort_index] = label_ppg[ib * chunk_len:(ib + 1) * chunk_len]

    def _reform_data_from_dict(self, data):
        sort_data = sorted(data.items(), key=lambda x: x[0])
        sort_data = [i[1] for i in sort_data]
        sort_data = torch.cat(sort_data, dim=0)
        return np.reshape(sort_data.cpu(), (-1))

    def _calculate_hr_metrics(self, predictions, labels, data_config):
        predict_hr_fft_all = []
        gt_hr_fft_all = []
        snr_all = []
        if data_config.PREPROCESS.LABEL_TYPE in ["Standardized", "Raw"]:
            diff_flag = False
        elif data_config.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            diff_flag = True
        else:
            raise ValueError("Unsupported label type in validation!")

        for index in predictions.keys():
            prediction = self._reform_data_from_dict(predictions[index])
            label = self._reform_data_from_dict(labels[index])
            video_frame_size = prediction.shape[0]
            if self.config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
                window_frame_size = int(self.config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * data_config.FS)
                if window_frame_size > video_frame_size:
                    window_frame_size = video_frame_size
            else:
                window_frame_size = video_frame_size

            for i in range(0, len(prediction), window_frame_size):
                pred_window = prediction[i:i+window_frame_size]
                label_window = label[i:i+window_frame_size]
                if len(pred_window) < 9:
                    continue
                gt_hr_fft, pred_hr_fft, snr = calculate_metric_per_video(
                    pred_window, label_window, diff_flag=diff_flag, fs=data_config.FS, hr_method='FFT')
                gt_hr_fft_all.append(gt_hr_fft)
                predict_hr_fft_all.append(pred_hr_fft)
                snr_all.append(snr)

        gt_hr_fft_all = np.array(gt_hr_fft_all)
        predict_hr_fft_all = np.array(predict_hr_fft_all)
        snr_all = np.array(snr_all)
        return {
            "MAE": np.mean(np.abs(predict_hr_fft_all - gt_hr_fft_all)),
            "RMSE": np.sqrt(np.mean(np.square(predict_hr_fft_all - gt_hr_fft_all))),
            "MAPE": np.mean(np.abs((predict_hr_fft_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100,
            "Pearson": np.corrcoef(predict_hr_fft_all, gt_hr_fft_all)[0][1],
            "SNR": np.mean(snr_all),
        }

    def _score_is_better(self, metric, score):
        return self.best_valid_score is None or (
            score > self.best_valid_score if metric in ["Pearson", "SNR"] else score < self.best_valid_score
        )

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        selection_metric = self.config.TRAIN.MODEL_SELECTION.METRIC
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()

            # Model Training
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels = batch[0].float(), batch[1].float()
                N, D, C, H, W = data.shape

                if self.config.TRAIN.AUG :
                    data,labels = self.data_augmentation(data,labels,batch[2],batch[3])

                data = data.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                pred_ppg, aux = self._unpack_model_output(self.model(data))
                pred_ppg = self._normalize_ppg(pred_ppg)

                loss = 0.0
                for ib in range(N):
                    loss = loss + self.criterion(
                        pred_ppg[ib],
                        labels[ib],
                        epoch,
                        self.config.TRAIN.DATA.FS,
                        self.diff_flag,
                        aux=self._slice_aux(aux, ib, N),
                    )
                loss = loss / N
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                tbar.set_postfix(loss=loss.item())
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss, valid_metrics = self.valid(
                    data_loader, return_metrics=selection_metric != "loss")
                print('validation loss: ', valid_loss)
                if selection_metric == "loss":
                    valid_score = valid_loss
                else:
                    valid_score = valid_metrics[selection_metric]
                    print("validation FFT metrics: MAE={MAE}, RMSE={RMSE}, MAPE={MAPE}, Pearson={Pearson}, SNR={SNR}".format(**valid_metrics))
                    print("validation selection score ({}): {}".format(selection_metric, valid_score))
                if self._score_is_better(selection_metric, valid_score):
                    self.min_valid_loss = valid_loss
                    self.best_valid_score = valid_score
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}, best_valid_score: {}".format(
                self.best_epoch, self.min_valid_loss, self.best_valid_score))  


    def valid(self, data_loader, return_metrics=False):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print('')
        print("===Validating===")
        valid_loss = []
        predictions = dict()
        labels = dict()
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid = valid_batch[0].to(self.device, non_blocking=True)
                labels_valid = valid_batch[1].to(self.device, non_blocking=True)
                N, D, C, H, W = data_valid.shape
                pred_ppg_valid = self._predict_ppg(data_valid)

                for ib in range(N):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS , self.config.VALID.DATA.FS , self.diff_flag)
                    valid_loss.append(loss.item())
                    valid_step += 1
                    vbar.set_postfix(loss=loss.item())
                if return_metrics:
                    self._store_batch_predictions(
                        predictions, labels, pred_ppg_valid, labels_valid, valid_batch, N, D)
        if return_metrics:
            return np.mean(np.asarray(valid_loss)), self._calculate_hr_metrics(
                predictions, labels, self.config.VALID.DATA)
        return np.mean(np.asarray(valid_loss)), None


    def test(self, data_loader):
        """ Model evaluation on the testing dataset."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        with torch.no_grad():
            predictions = dict()
            labels = dict()
            for _, test_batch in enumerate(data_loader['test']):
                batch_size = test_batch[0].shape[0]
                chunk_len = self.chunk_len
                data_test = test_batch[0].to(self.config.DEVICE, non_blocking=True)
                labels_test = test_batch[1].to(self.config.DEVICE, non_blocking=True)
                pred_ppg_test = self._predict_ppg(data_test)
                self._store_batch_predictions(
                    predictions, labels, pred_ppg_test, labels_test, test_batch, batch_size, chunk_len)
            print(' ')
            calculate_metrics(predictions, labels, self.config)


    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)


    def data_augmentation(self,data,labels,index1,index2):
        N, D, C, H, W = data.shape
        data_aug = data.clone()
        labels_aug = labels.clone()
        rand1_vals = np.random.random(N)
        rand2_vals = np.random.random(N)
        for idx in range(N):
            index = index1[idx] + index2[idx]
            rand1 = rand1_vals[idx]
            if rand1 < 0.5 :
                if index in self.data_dict:
                    gt_hr_fft = self.data_dict[index]
                else:
                    gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
                    self.data_dict[index] = gt_hr_fft
                    
                if gt_hr_fft > 90: 
                    max_start = max(0, D - (D // 2 + 1))
                    rand3 = random.randint(0, max_start)
                    even_indices = torch.arange(0, D, 2)
                    odd_indices = torch.arange(1, D, 2)
                    data_aug[idx, even_indices, :, :, :] = data[idx, rand3 + even_indices // 2, :, :, :]
                    labels_aug[idx, even_indices] = labels[idx, rand3 + even_indices // 2]
                    data_aug[idx, odd_indices, :, :, :] = (data[idx, rand3 + odd_indices // 2, :, :, :] + data[idx, rand3 + (odd_indices // 2) + 1, :, :, :]) / 2
                    labels_aug[idx, odd_indices] = (labels[idx, rand3 + odd_indices // 2] + labels[idx, rand3 + (odd_indices // 2) + 1]) / 2
                elif gt_hr_fft < 75 :
                    data_downsampled = data[idx, ::2, :, :, :]
                    labels_downsampled = labels[idx, ::2]
                    data_aug[idx] = torch.cat([data_downsampled, data_downsampled], dim=0)[:D]
                    labels_aug[idx] = torch.cat([labels_downsampled, labels_downsampled], dim=0)[:D]
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
            if rand2_vals[idx] < 0.5:
                data_aug[idx] = torch.flip(data_aug[idx], dims=[3])
        return data_aug, labels_aug
