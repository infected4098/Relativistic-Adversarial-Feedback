import warnings
import scipy
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import easydict
from tqdm import tqdm
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torchaudio.functional import frechet_distance as frechet
import time
from quality_model.scoreq import Scoreq
from bigvgan_utils import AttrDict, build_env, summarize_model, load_checkpoint, save_checkpoint, scan_checkpoint
from dataset import MelDataset
import matplotlib.pyplot as plt
from eval_metrics import evaluate_obj
from bigvgan import BigVGAN

warnings.simplefilter(action='ignore', category=FutureWarning)


def prefix_load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    prefix = "module."
    for key in state_dict["generator"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = state_dict["generator"][key]

    print("Complete.")
    return new_state_dict

def clean_state_dict(state_dict, suffixes=("_g", "_v")):

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        for suffix in suffixes:
            if key.endswith(suffix):
                key = key[: -len(suffix)]  # Remove the suffix
        cleaned_state_dict[key] = value
    return cleaned_state_dict


def evaluate_single(a, eval_file_lsts, h, is_eval_obj=True, is_sample=True, is_output=True):
    assert a.sampling_rate == h.sampling_rate, "The sampling rate for every config should be same"
    # Hyperparameters setting
    module_path_ = a.module_path
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda')
    suffix = "_generated_samples"
    # Output path should be same as the module path but not the final dir

    output_path = module_path_.split("/")[:-1]
    output_path = "/".join(output_path)
    data_name = module_path_.split("/")[-1] + a.dataset_name
    output_path = os.path.join(output_path, data_name)
    output_path = os.path.join(output_path, "generated_samples")
    #output_path = os.path.join(a.output_path, suffix)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    # Model initialization
    
    if a.model_name in ["BIGVGAN", "BIGVGAN_BASE"]:
        generator = BigVGAN(h, use_cuda_kernel=False).to(device)
        try:
            state_dict_g = load_checkpoint(module_path_, device)
            generator.load_state_dict(state_dict_g["generator"])
        except:
            state_dict_g = prefix_load_checkpoint(module_path_, device)
        #state_dict_g = clean_state_dict(state_dict_g, suffixes=("_g", "_v"))
            generator.load_state_dict(state_dict_g)  
        generator.remove_weight_norm()
      
    elif a.model_name == "VOCOS_PT":
        pass
    elif a.model_name == "VOCOS_TRAIN":
        pass
    elif a.model_name == "PERIODWAVE":
        pass
    elif a.model_name == "WAVEFM":
        pass

    # Dataset

    if a.dataset_name == "LIBRITTS":
        filelist =  []
        with open(eval_file_lsts.LIBRITTS, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    filelist.append(line)
                  
    elif a.dataset_name == "LJSPEECH":
        filelist = []
        with open(eval_file_lsts.LJSPEECH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    filelist.append(line)

    elif a.dataset_name == "MUSDB18HQ":
        filelist =  []
        idx_lst = []
        with open(eval_file_lsts.MUSDB18HQ, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(',')  # Remove leading/trailing whitespace
                if line:  # Skip empty lines
                    filelist.append(line[0])
                    idx_lst.append(int(line[1]))  

    elif a.dataset_name == "UR":
        filelist = []
        with open(eval_file_lsts.UR, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    filelist.append(line)
                  
    elif a.dataset_name == "DEEPLY":
        filelist = []
        with open(eval_file_lsts.DEEPLY, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    filelist.append(line)


    dataset_for_mos = MelDataset(
        filelist,
        h,
        h.segment_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        False,
        False,
        fmax_loss=h.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
        is_seen=True,
    )


    generator.eval()
    torch.cuda.empty_cache()


    # MOS Predictor
    utmos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)

    utmos_score  = 0 
    score_matrix = np.zeros([int(len(dataloader_for_mos)), 2])
    print("score matrix shape is :", score_matrix.shape)
    print("Sample Generation...")
    with torch.no_grad():

        inference_time = 0
        audio_time = 0
        with torch.no_grad():
            for j, batch_mos in tqdm(enumerate(dataloader_for_mos)):
                x_mos, y_mos, _, y_mel = batch_mos # [1, num_mels, sequence_length]
                start = time.perf_counter()
                pred_mos = generator(x_mos.to(device)) # [1, 1, sequence_length]
                end = time.perf_counter()
                inference_t = end - start
                audio_time += pred_mos.shape[-1] / h.sampling_rate
                gt_mos = y_mos.unsqueeze(1)

                if is_output:
                    y_np = gt_mos.cpu().detach().numpy().reshape(-1)
                    y_np = y_np / np.max(np.abs(y_np))
                    y_np = (y_np * 32767).astype(np.int16)

                    y_g_hat_np = pred_mos.cpu().detach().numpy().reshape(-1)
                    y_g_hat_np = y_g_hat_np / np.max(np.abs(y_g_hat_np))
                    y_g_hat_np = (y_g_hat_np * 32767).astype(np.int16)

                    scipy.io.wavfile.write(os.path.join(output_path, f"gt_sample_{j}.wav"), h.sampling_rate, y_np)
                    scipy.io.wavfile.write(os.path.join(output_path, f"pred_sample_{j}.wav"), h.sampling_rate, y_g_hat_np)
                    
                
            if is_output:
                print(f"The output samples are in {output_path}")

            if is_eval_obj:
                print("Complete \n")
                score_matrix, scoreq_score, utmos_score, pitch_rmse, periodicity_rmse, f1 = evaluate_obj(score_matrix, output_path, device, gpu=0, model_name=a.model_name, sampling_rate=a.sampling_rate)
                utmos_score_mean = utmos_score / (j+1)
                print("\n Utmos of the model is : ", utmos_score_mean)
                score = np.nanmean(score_matrix, axis=0) # Ignore nan values
                print(f": MRSTFT: {score[0]}, PESQ: {score[1]}, Periodicity_RMSE: {periodicity_rmse}, F1: {f1} \n")
                print(f"The evaluation on {a.model_name} is over...")
                print("Complete...")

                return score, pitch_rmse, periodicity_rmse, f1, scoreq_score_mean, utmos_score_mean



def main():
    print('Initializing Training Process..')
    a = easydict.EasyDict({
    "group_name" : None,
    "input_mels_dir": 'ft_dataset',
    "fine_tuning": False,
    "experiment_name": "NULL",
    "module_path": "./raf_bigvganbase_1M",  
    "dataset_name": "LIBRITTS",
    "data_mode": "VAL",
    "libritts_dir": "./LibriTTS",
    'config_bigvgan_base': './config.json',
    "model_name": "BIGVGAN_BASE",
    "sampling_rate": 24000
    })
    eval_file_lsts = easydict.EasyDict({
        "LJSPEECH": './filelists/ljspeech.txt',
        "LIBRITTS": "./filelists/LIBRITTS.txt",
        "LIBRITTS_MOS": "./filelists/LIBRITTS_MOS.txt",
        "MUSDB18HQ_MOS": "./filelists/MUSDB.txt",
        "UR": "./filelists/ur.txt",
        "UR_MOS":'./filelists/ur_mos.txt',
        "DEEPLY": "./filelists/deeply.txt",
        "MUSDB18HQ": "./filelists/MUSDB_vocal.txt",
        "DEEPLY_MOS": "./filelists/deeply_mos.txt"
    })

    
    if a.model_name == "HIFIGAN":
        with open(a.config_hifigan) as f:
            data = f.read()         
    elif a.model_name == "BIGVGAN_BASE":
        with open(a.config_bigvgan_base) as f:
            data = f.read()


    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    utmos, mrstft, pesq, pitch, periodicity, f1score = [], [], [], [], [], []
    score_matrix, pitch_rmse, periodicity_rmse, f1, scoreq_score_mean, utmos_score_mean = evaluate_single(a, eval_file_lsts, h, is_eval_obj=True, is_sample=False, is_output=True)
    utmos.append(utmos_score_mean)
    mrstft.append(score_matrix[0])
    pesq.append(score_matrix[1])
    f1score.append(f1)
    pitch.append(pitch_rmse)
    periodicity.append(periodicity_rmse)

    print("MRSTFT score: ", np.array(mrstft).mean(), "PESQ: ", np.array(pesq).mean(), 
        "Pitch:", np.array(pitch).mean(), "Periodicity :", np.array(periodicity).mean(), "F1 score: ", np.array(f1score).mean(), 
        "UTMOS score: ", np.array(utmos).mean())    

if __name__ == '__main__':
  # a = Args()
  main()
