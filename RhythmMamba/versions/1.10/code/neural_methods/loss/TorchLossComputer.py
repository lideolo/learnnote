'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import pdb
import torch.nn as nn
from evaluation.post_process import calculate_hr , calculate_psd

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss

class Hybrid_Loss(nn.Module): 
    def __init__(
        self,
        time_weight=0.2,
        freq_ce_weight=1.0,
        freq_kl_weight=0.0,
        freq_std=3.0,
        roi_phase_weight=0.0,
        harmonic_weight=0.0,
        aux_hr_weight=0.0,
        aux_hr_std=3.0,
    ):
        super(Hybrid_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
        self.time_weight = time_weight
        self.freq_ce_weight = freq_ce_weight
        self.freq_kl_weight = freq_kl_weight
        self.freq_std = freq_std
        self.roi_phase_weight = roi_phase_weight
        self.harmonic_weight = harmonic_weight
        self.aux_hr_weight = aux_hr_weight
        self.aux_hr_std = aux_hr_std

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag, aux=None):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_Fre , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=self.freq_std)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = self.time_weight * loss_time + self.freq_ce_weight * loss_Fre + self.freq_kl_weight * loss_distribution_kl
        if aux is not None and self.roi_phase_weight > 0:
            loss = loss + self.roi_phase_weight * TorchLossComputer.ROI_phase_consistency_loss(
                aux.get("roi_tokens"), aux.get("roi_weights"), labels, diff_flag, FS, aux.get("token_fs"))
        if self.harmonic_weight > 0:
            loss = loss + self.harmonic_weight * TorchLossComputer.Harmonic_consistency_loss(
                pred_ppg.squeeze(-1), labels.squeeze(-1), diff_flag, FS)
        if aux is not None and self.aux_hr_weight > 0:
            loss = loss + self.aux_hr_weight * TorchLossComputer.Auxiliary_hr_distribution_loss(
                aux.get("hr_logits"), labels.squeeze(-1), diff_flag, FS, std=self.aux_hr_std)
        return loss
    
class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()
    def forward(self, pred_ppg, labels, epoch, FS, diff_flag):    
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))    
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss

class PhysFormer_Loss(nn.Module): 
    def __init__(self):
        super(PhysFormer_Loss,self).__init__()
        self.criterion_Pearson = Neg_Pearson()

    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):       
        loss_rPPG = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1) , diff_flag = diff_flag , Fs = FS, std=1.0)
        if torch.isnan(loss_rPPG) : 
           loss_rPPG = 0
        if epoch >30:
            a = 1.0
            b = 5.0
        else:
            a = 1.0
            b = 1.0*math.pow(5.0, epoch/30.0)

        loss = a * loss_rPPG + b * (loss_distribution_kl + loss_CE)
        return loss
    
class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        device = output.device
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float, device=device), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor).to(device), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).to(device)
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).to(device=device, dtype=torch.float32)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz
        
        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator
        
        
    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

        
    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx
    
    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_pred, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        device = inputs.device
        bpm_range = torch.arange(45, 150, dtype=torch.float).to(device)
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(45, 150)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)

        hr_gt = torch.tensor(hr_gt-45).view(1).type(torch.long).to(device)
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)

    @staticmethod
    def Auxiliary_hr_distribution_loss(hr_logits, target, diff_flag, Fs, std):
        if hr_logits is None:
            return target.new_tensor(0.0)

        _, hr_gt = calculate_hr(target.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
        hr_gt = int(round(hr_gt))
        hr_gt = max(45, min(150, hr_gt))
        target_index = torch.tensor([hr_gt - 45], dtype=torch.long, device=hr_logits.device)
        ce_loss = F.cross_entropy(hr_logits.view(1, -1), target_index)

        bpm_range = torch.arange(45, 151, dtype=torch.float, device=hr_logits.device)
        target_distribution = [normal_sampling(hr_gt, i, std) for i in range(45, 151)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(hr_logits.device)
        target_distribution = target_distribution / target_distribution.sum()
        log_prob = F.log_softmax(hr_logits.view(-1), dim=0)
        kl = F.kl_div(log_prob, target_distribution, reduction="sum")
        return ce_loss + kl

    @staticmethod
    def ROI_phase_consistency_loss(roi_tokens, roi_weights, target, diff_flag, Fs, token_fs):
        if roi_tokens is None or roi_weights is None or token_fs is None:
            return target.new_tensor(0.0)

        _, hr_gt = calculate_hr(target.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
        token_fs = float(token_fs.detach().cpu().item()) if torch.is_tensor(token_fs) else float(token_fs)
        roi_ppg = roi_tokens.mean(dim=-1)
        roi_ppg = roi_ppg - roi_ppg.mean(dim=0, keepdim=True)
        spectrum = torch.fft.rfft(roi_ppg, dim=0, norm="ortho")
        if spectrum.shape[0] <= 2:
            return roi_tokens.new_tensor(0.0)

        unit_hz = token_fs / roi_ppg.shape[0]
        base_idx = int(round((hr_gt / 60.0) / unit_hz))
        base_idx = max(1, min(spectrum.shape[0] - 1, base_idx))
        base = spectrum[base_idx]
        phase = base / (torch.abs(base) + 1e-6)
        weights = roi_weights / (roi_weights.sum() + 1e-6)
        mean_phase = torch.sum(weights * phase)
        return 1.0 - torch.abs(mean_phase)

    @staticmethod
    def Harmonic_consistency_loss(inputs, target, diff_flag, Fs):
        _, hr_gt = calculate_hr(inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs)
        signal = inputs.view(-1)
        signal = signal - signal.mean()
        spectrum = torch.abs(torch.fft.rfft(signal, dim=0, norm="ortho"))
        if spectrum.shape[0] <= 3:
            return inputs.new_tensor(0.0)

        freqs = torch.fft.rfftfreq(signal.shape[0], d=1.0 / Fs).to(inputs.device)
        unit_hz = Fs / signal.shape[0]
        base_idx = int(round((hr_gt / 60.0) / unit_hz))
        base_idx = max(1, min(spectrum.shape[0] - 1, base_idx))
        harmonic_idx = min(spectrum.shape[0] - 1, base_idx * 2)
        drift_mask = (freqs > 0) & (freqs < 0.5)
        drift_amp = spectrum[drift_mask].max() if torch.any(drift_mask) else inputs.new_tensor(0.0)
        base_amp = spectrum[base_idx].clamp_min(1e-6)
        harmonic_ratio = spectrum[harmonic_idx] / base_amp
        drift_ratio = drift_amp / base_amp
        return F.relu(drift_ratio - 0.8) ** 2 + 0.25 * F.relu(harmonic_ratio - 1.5) ** 2
    
    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_pred, psd_gt = calculate_psd(inputs.detach().cpu(), target.detach().cpu() , diff_flag = diff_flag , fs=Fs)
        device = inputs.device
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(device)
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)
        return kl_loss(pred_distribution , target_distribution)
