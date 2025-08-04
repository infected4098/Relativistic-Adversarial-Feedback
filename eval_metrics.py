import auraloss
import numpy as np
from tqdm import tqdm
from scoreq import Scoreq
import torch
import librosa
import os
import numpy as np
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from utils import downsample_speech_cuda
import torchcrepe
import functools
import torchaudio
import warnings
from tqdm import tqdm
import re
from pesq import pesq


def scoreq_score_val_ds(model, pred, sr):
  pred_16k = downsample_speech_cuda(pred, sr, 16000).unsqueeze(1) #[B, 1, sequence_length]
  pred_score = model.predict(test_path = pred_16k[0])
  return pred_score.item()

class Pitch:

    def __init__(self):
        self.threshold = torchcrepe.threshold.Hysteresis()
        self.reset()
    
    def __call__(self):
        pitch_rmse = torch.sqrt(self.pitch_total / self.voiced)
        periodicity_rmse = torch.sqrt(self.periodicity_total / self.count)
        precision = \
            self.true_positives / (self.true_positives + self.false_positives)
        recall = \
            self.true_positives / (self.true_positives + self.false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return {
            'pitch': pitch_rmse.item(),
            'periodicity': periodicity_rmse.item(),
            'f1': f1.item(),
            'precision': precision.item(),
            'recall': recall.item()}

    def reset(self):
        self.count = 0
        self.voiced = 0
        self.pitch_total = 0.
        self.periodicity_total = 0.
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
    
    def update(self, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
        # Threshold
        true_threshold = self.threshold(true_pitch, true_periodicity)
        pred_threshold = self.threshold(pred_pitch, pred_periodicity)
        true_voiced = ~torch.isnan(true_threshold)
        pred_voiced = ~torch.isnan(pred_threshold)

        # Update periodicity rmse
        self.count += true_pitch.shape[1]
        self.periodicity_total += (true_periodicity - pred_periodicity).pow(2).sum()

        # Update pitch rmse
        voiced = true_voiced & pred_voiced
        self.voiced += voiced.sum()
        difference_cents = 1200 * (torch.log2(true_pitch[voiced]) - 
                                   torch.log2(pred_pitch[voiced]))
        self.pitch_total += difference_cents.pow(2).sum()
        
        # Update voiced/unvoiced precision and recall
        self.true_positives += (true_voiced & pred_voiced).sum()
        self.false_positives += (~true_voiced & pred_voiced).sum()
        self.false_negatives += (true_voiced & ~pred_voiced).sum()

def from_audio(audio, sample_rate=24000, gpu=None):
    """Preprocess pitch from audio"""
    # Target number of frames
    target_length = audio.shape[1] // 2048
    
    # Resample
    if sample_rate != torchcrepe.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate,
                                                   torchcrepe.SAMPLE_RATE)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
    
    # Resample hopsize
    hopsize = int(256 * (torchcrepe.SAMPLE_RATE / sample_rate))

    # Pad
    padding = int((1024 - hopsize) // 2)
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect').squeeze(0)

    # Estimate pitch
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        fmin=50,
        fmax=550,
        model='full',
        return_periodicity=True,
        batch_size=50,
        device='cpu' if gpu is None else f'cuda:{gpu}',
        pad=False)

    # Set low energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False)

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity

def find_wav_files(directory):
    """Walk through a directory and find all .wav files, aligning them by numeric order."""
    gt_files = []
    pred_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                full_path = os.path.join(root, file)
                if file.startswith('gt_'):
                    gt_files.append(full_path)
                elif file.startswith('pred_'):
                    pred_files.append(full_path)

    # Sort files by the numeric part of their filenames
    def extract_number(file_path):
        match = re.search(r'_(\d+)', os.path.basename(file_path))
        return int(match.group(1)) if match else float('inf')

    gt_files.sort(key=extract_number)
    pred_files.sort(key=extract_number)

    return gt_files, pred_files

def pitch(directory, gpu=None, sample_rate=22050):
    pitch_rmse, periodicity_rmse, f1 = 0, 0, 0
    # Setup metrics
    batch_metrics = Pitch()
    metrics = Pitch()
    gt_files, pred_files = find_wav_files(directory)
    file_results = {}
    # Pitch and periodicity extraction
    pitch_fn = functools.partial(
        from_audio,
        gpu=gpu)
    
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    for i, (gt, pred) in tqdm(enumerate(zip(gt_files, pred_files))):
        
        # Load audio
        gt_audio = torchaudio.load(gt)[0].to(torch.float32)
        pred_audio = torchaudio.load(pred)[0].to(torch.float32)
        # I want to pad the short audio to longest audio between gt_audio, pred_audio.
        if gt_audio.shape[1] < pred_audio.shape[1]:
            gt_audio = torch.nn.functional.pad(gt_audio, (0, pred_audio.shape[1] - gt_audio.shape[1]), mode='constant', value=0)
        elif gt_audio.shape[1] > pred_audio.shape[1]:
            pred_audio = torch.nn.functional.pad(pred_audio, (0, gt_audio.shape[1] - pred_audio.shape[1]), mode='constant', value=0)
        # Get true pitch
        true_pitch, true_periodicity = pitch_fn(gt_audio, gpu=gpu, sample_rate=sample_rate)
        # Load vocoded audio

        # I want to pad 

        # Estimate pitch
        pred_pitch, pred_periodicity = pitch_fn(pred_audio, gpu=gpu, sample_rate=sample_rate)
        
        # Get metrics for this file
        metrics.reset()
        metrics.update(
            true_pitch,
            true_periodicity,
            pred_pitch,
            pred_periodicity)
        file_results[i] = metrics()

    for j in range(len(file_results)):
        pitch_rmse += file_results[j]['pitch']
        periodicity_rmse += file_results[j]['periodicity']
        f1 += file_results[j]['f1']
    pitch_rmse /= len(file_results)
    periodicity_rmse /= len(file_results)
    f1 /= len(file_results)

    return pitch_rmse, periodicity_rmse, f1

def zero_pad(sequence, ref_sequence):
    if sequence.shape[2] < ref_sequence.shape[2]:
        diff = sequence.shape[2] - ref_sequence.shape[2]
        pred_output = torch.zeros_like(ref_sequence)
        pred_output[:, :, :sequence.shape[2]] = sequence
        gt_output = ref_sequence

    elif sequence.shape[2] == ref_sequence.shape[2]:
        diff = sequence.shape[2] - ref_sequence.shape[2]
        pred_output = sequence
        gt_output = ref_sequence

    else:
        diff = sequence.shape[2] - ref_sequence.shape[2]
        gt_output = torch.zeros_like(sequence)
        gt_output[:, :, :ref_sequence.shape[2]] = ref_sequence
        pred_output = sequence
    return pred_output, gt_output

def eval_obj(input_tensor, gt_tensor, device, model_name = "RAF", sampling_rate = 24000):
    mrstft = auraloss.freq.MultiResolutionSTFTLoss()
    # Initializing the score metrics
    mrstft_loss = 0
    pesq_score = 0
    input_tensor, gt_tensor = zero_pad(input_tensor, gt_tensor)
    wb_pesq = PerceptualEvaluationSpeechQuality(16000, 'wb')

    # Minimal amounts for computing MRSTFT
    if input_tensor.shape[2] <= 8192: 
        print(f"Sequence length is too short for appropriate comparison. Skipping the evaluation of tensor of shape {input_tensor.shape}")
        return np.zeros([2])
    else:
        # input_tensor.shape, gt_tensor.shape = [1, 1, sequence_length]
        mrstft_loss += mrstft(input_tensor.cpu(), gt_tensor.cpu())
        # [sequence_length,]
        input_tensor = input_tensor.reshape(-1)
        gt_tensor = gt_tensor.reshape(-1)
        # PESQ can possibly miss the speech fragments
        try:
            input_tensor_16k = downsample_speech_cuda(input_tensor, sampling_rate, 16000)
            gt_tensor_16k = downsample_speech_cuda(gt_tensor, sampling_rate, 16000)
            #pesq_score += wb_pesq(input_tensor_16k, gt_tensor_16k).item()
            pesq_score += pesq(16000, gt_tensor_16k.cpu().numpy().reshape(-1), input_tensor_16k.cpu().numpy().reshape(-1), 'wb')
        except:
            print(f"PESQ Score Unobtainable for model {model_name}")
            pesq_score = np.nan

        input_tensor = input_tensor.cpu().numpy()
        gt_tensor = gt_tensor.cpu().numpy()

        return np.array([mrstft_loss, pesq_score])



def evaluate_obj(score_matrix, directory, device, gpu, model_name = "RAF", sampling_rate = 24000):
    pitch_rmse, periodicity_rmse, f1 = 0, 0, 0
    # Setup metrics
    batch_metrics = Pitch()
    metrics = Pitch()
    gt_files, pred_files = find_wav_files(directory)
    file_results = {}
    # Pitch and periodicity extraction
    pitch_fn = functools.partial(
        from_audio,
        gpu=gpu)
    
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)

    scoreq_val = Scoreq(data_domain='natural', device = device, mode='nr')
    utmos_score = 0
    scoreq_score = 0
    gt_files, pred_files = find_wav_files(directory)
    for i, (gt, pred) in tqdm(enumerate(zip(gt_files, pred_files))):
    
        # Load audio
        gt_audio = torch.from_numpy(librosa.load(gt)[0]).to(torch.float32).to(device).unsqueeze(0)
        pred_audio = torch.from_numpy(librosa.load(pred)[0]).to(torch.float32).to(device).unsqueeze(0)
        score_matrix[i] = eval_obj(pred_audio.unsqueeze(0), gt_audio.unsqueeze(0), device, model_name = model_name, sampling_rate = sampling_rate)
        utmos_score += utmos_predictor(pred_audio.squeeze(1), sr=sampling_rate).item()
        scoreq_score += scoreq_score_val_ds(scoreq_val, pred_audio.squeeze(1).detach(), sr=sampling_rate)
        pred_audio, gt_audio = zero_pad(pred_audio.unsqueeze(0), gt_audio.unsqueeze(0))
        pred_audio = pred_audio.squeeze(0)
        gt_audio = gt_audio.squeeze(0)
        """if gt_audio.shape[1] < pred_audio.shape[1]:
            gt_audio = torch.nn.functional.pad(gt_audio, (0, pred_audio.shape[1] - gt_audio.shape[1]), mode='constant', value=0)
        elif gt_audio.shape[1] > pred_audio.shape[1]:
            pred_audio = torch.nn.functional.pad(pred_audio, (0, gt_audio.shape[1] - pred_audio.shape[1]), mode='constant', value=0)"""
        # Get true pitch
        true_pitch, true_periodicity = pitch_fn(gt_audio, gpu=gpu, sample_rate=sampling_rate)
        # Load vocoded audio

        # I want to pad 

        # Estimate pitch
        pred_pitch, pred_periodicity = pitch_fn(pred_audio, gpu=gpu, sample_rate=sampling_rate)
        
        # Get metrics for this file
        metrics.reset()
        metrics.update(
            true_pitch,
            true_periodicity,
            pred_pitch,
            pred_periodicity)
        file_results[i] = metrics()
    invalid = 0
    pitch_array = np.array([file_results[j]['pitch'] for j in range(len(file_results))])
    periodicity_array = np.array([file_results[j]['periodicity'] for j in range(len(file_results))])
    f1_array = np.array([file_results[j]['f1'] for j in range(len(file_results))])
    '''    for j in range(len(file_results)):
        # file_results is a matrix with keys 'pitch', 'periodicity', 'f1'. If any of the components are nan, skip the file.
        if any(np.isnan(list(file_results[j].values()))):
            print(f"Skipping file {j} due to NaN values in results.")
            invalid += 1
            continue
        else:
            pitch_rmse += file_results[j]['pitch']
            periodicity_rmse += file_results[j]['periodicity']
            f1 += file_results[j]['f1']
    print(invalid)
    pitch_rmse /= (len(file_results) - invalid)
    periodicity_rmse /= (len(file_results) - invalid)
    f1 /= (len(file_results) - invalid)'''
    pitch_rmse = np.nanmean(pitch_array)
    periodicity_rmse = np.nanmean(periodicity_array)
    f1 = np.nanmean(f1_array)
    return score_matrix, scoreq_score, utmos_score, pitch_rmse, periodicity_rmse, f1
