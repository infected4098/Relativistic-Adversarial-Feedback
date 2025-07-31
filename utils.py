import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
from torch.autograd import grad
from torchaudio.transforms import Resample
import torch.nn.functional as F
MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging

def dydg_asym(disc_real_outputs, disc_generated_outputs, big="dr"):
   smallds = []
   if big == "dr":
      for dr, dg in zip(disc_real_outputs, disc_generated_outputs): # [B, 3, sequence_length]
        smalld = torch.mean(F.softplus(dr-dg), dim=-1)
        #smalld = torch.mean(dr-dg, dim=-1)
        smallds.append(smalld)
   elif big == "dg":
      for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        smalld = torch.mean(F.softplus(dg-dr), dim=-1)
        #smalld = torch.mean(dg-dr, dim=-1)
        smallds.append(smalld)
   return torch.stack(smallds, dim=0)


def repeat_qydiffqg(qyqg, N):
  return qyqg.unsqueeze(0).repeat(N, 1, 1)


def zero_centered_gradient_penalty(model, real_samples, fake_samples, device="cuda"):

    real_samples = real_samples.detach().to(device).requires_grad_(True)
    fake_samples = fake_samples.detach().to(device).requires_grad_(True)
    y_d_rs, y_d_gs, _, _ = model(real_samples, fake_samples)
    ### R1
    gradient_penalties_r1 = []
    for y_d_r in y_d_rs:
        grads = grad(
            outputs=y_d_r,  # Ground truth
            inputs=real_samples,
            create_graph=True,
            grad_outputs=torch.ones_like(y_d_r).to(device),
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm_r1 = grads.square().mean() # [B, 1, sequence_length]
        gradient_penalties_r1.append(grad_norm_r1)

    total_gradient_penalty_r1 = torch.mean(torch.stack(gradient_penalties_r1))

    ### R2
    gradient_penalties_r2 = []
    for y_d_g in y_d_gs:
        grads = grad(
            outputs=y_d_g,  # Prediction
            inputs=fake_samples,
            create_graph=True,
            grad_outputs=torch.ones_like(y_d_g).to(device),
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_norm_r2 = grads.square().mean()

        gradient_penalties_r2.append(grad_norm_r2)

    total_gradient_penalty_r2 = torch.mean(torch.stack(gradient_penalties_r2))

    return total_gradient_penalty_r1, total_gradient_penalty_r2

def downsample_speech_cuda(signal: torch.Tensor, original_sample_rate: int, target_sample_rate: int) -> torch.Tensor:

    device = signal.device
    if original_sample_rate == target_sample_rate:
        return signal

    resampler = Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate).to(device)
    downsampled_signal = resampler(signal)
    return downsampled_signal

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

def prefix_load_checkpoint_discriminator(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    new_state_dict_mpd = {}
    new_state_dict_mrd = {}
    prefix = "module."
    for key in state_dict["mpd"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict_mpd[new_key] = state_dict["mpd"][key]
    for key in state_dict["mrd"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict_mrd[new_key] = state_dict["mrd"][key]     

    print("Complete.")
    return new_state_dict_mpd, new_state_dict_mrd, state_dict["steps"], state_dict["epoch"]
