# The train code is adapted from https://github.com/sh-lee-prml/PeriodWave and https://github.com/NVIDIA/BigVGAN.

# Implementation of Relativistic Adversarial Feedback (RAF) training framework for GAN-based vocoders

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
import easydict
from dataset import get_dataset_filelist, MelDataset, mel_spectrogram, MAX_WAV_VALUE
from bigvgan import BigVGAN
from discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator, \
    feature_loss, generator_loss, discriminator_loss
from quality_model.scoreq import Scoreq
from utils import downsample_speech_cuda, zero_centered_gradient_penalty, discriminator_gap, repeat_quality_gap, prefix_load_checkpoint, prefix_load_checkpoint_discriminator
from bigvgan_utils import AttrDict, build_env, load_checkpoint, save_checkpoint, scan_checkpoint
from quality_model.wavlm import WavLM, WavLMConfig
import torchaudio
import random
from msstft import QMultiScaleSTFTLoss
from audiotools import AudioSignal
import torchaudio
import random
import torch.distributed as dist

os.environ['MASTER_ADDR'] = 'localhost'
torch.backends.cudnn.benchmark = True
steps = 0

# RAF Gamma Scheduler for zero-centered gradient penalty coefficient
class RAFGammaScheduler():
    """
    Scheduler for RAF gamma parameter that controls gradient penalty regularization strength.
    Gamma is linearly decreased from start_gamma to end_gamma over num_epochs.
    """
    def __init__(self, start_gamma, end_gamma, num_epochs):
        self.start_gamma = start_gamma
        self.end_gamma = end_gamma
        self.num_epochs = num_epochs
        self.gamma_values = self._compute_gamma_schedule()
    
    def _compute_gamma_schedule(self):
        return torch.linspace(self.start_gamma, self.end_gamma, self.num_epochs)
    
    def get_gamma(self, epoch):
        if epoch >= self.num_epochs:
            return self.end_gamma
        return self.gamma_values[epoch].item()

# Utility function to count model parameters
def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def run(rank, n_gpus, a, hps):
    """
    Main training loop implementing RAF (Relativistic Adversarial Feedback) framework.
    RAF minimizes SSL model-aided discriminator gaps through relativistic feedback.
    """
    # Initialize RAF gamma scheduler - gamma is linearly decreased to 80% of original value
    start_gamma = 0.1
    end_gamma = 0.08
    raf_gamma_scheduler = RAFGammaScheduler(start_gamma, end_gamma, a.training_epochs)

    global steps

    # Initialize distributed training if using multiple GPUs
    if n_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    
    torch.manual_seed(hps.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # Collecting filelists for training and validation
    training_filelist, validation_filelist = get_dataset_filelist(a)

    # Training dataset configuration
    trainset = MelDataset(
        training_filelist,
        hps,
        hps.segment_size,  # RAF uses longer segments (24,576) for better quality gap estimation
        hps.n_fft,
        hps.num_mels,
        hps.hop_size,
        hps.win_size,
        hps.sampling_rate,
        hps.fmin,
        hps.fmax,
        shuffle=False if hps.num_gpus > 1 else True,
        fmax_loss=hps.fmax_for_loss,
        device=device,
        fine_tuning=a.fine_tuning,
        base_mels_path=a.input_mels_dir,
        is_seen=True,
    )
    
    train_sampler = DistributedSampler(trainset, rank = rank) if n_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=hps.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=hps.batch_size, pin_memory=True, drop_last=True)

    # Validation dataset configuration
    if rank == 0:
        validset = MelDataset(
            validation_filelist,
            hps,
            hps.segment_size,
            hps.n_fft,
            hps.num_mels,
            hps.hop_size,
            hps.win_size,
            hps.sampling_rate,
            hps.fmin,
            hps.fmax,
            False,
            False,
            fmax_loss=hps.fmax_for_loss,
            device=device,
            fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir,
            is_seen=True,
        )
        validation_loader = DataLoader(
            validset,
            num_workers=1,
            shuffle=False,
            sampler=None,
            batch_size=1,
            pin_memory=True,
            drop_last=True,
        )

        # Initialize Weights & Biases logging
        wandb.init(project=f"{a.experiment_name}", resume="allow")
        wandb.run.name = ""
        wandb.config.update(hps)

    # Initialize RAF Generator (BigVGAN architecture)
    raf_generator = BigVGAN(hps).cuda()

    # Initialize RAF Discriminators with extended output for quality gap components
    raf_mpd = MultiPeriodDiscriminator(hps).cuda()

    # Choose discriminator architecture based on configuration
    if hps.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print("[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator")
        raf_mrd = MultiBandDiscriminator(hps).cuda()
    elif hps.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print("[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator")
        raf_mrd = MultiScaleSubbandCQTDiscriminator(hps).cuda()
    else:  # Fallback to original MRD in BigVGAN-v1
        raf_mrd = MultiResolutionDiscriminator(hps, device).cuda()

    # Print model parameter counts
    if rank == 0:
        num_param = get_param_num(raf_generator)
        print('Number of Parameters for RAF Generator:', num_param)
        print("Number of Parameters for RAF MPD:  ", get_param_num(raf_mpd))
        print("Number of Parameters for RAF MRD:  ", get_param_num(raf_mrd))

    # Initialize RAF optimizers with AdamW
    raf_optimizer_g = torch.optim.AdamW(raf_generator.parameters(), hps.learning_rate, betas=[hps.adam_b1, hps.adam_b2])
    raf_optimizer_d = torch.optim.AdamW(itertools.chain(raf_mrd.parameters(), raf_mpd.parameters()),
                                hps.learning_rate, betas=[hps.adam_b1, hps.adam_b2])

    # Load checkpoints if available
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        try:
            state_dict_g = load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            raf_generator.load_state_dict(state_dict_g['generator'])
            raf_mpd.load_state_dict(state_dict_do['mpd'])
            raf_mrd.load_state_dict(state_dict_do['mrd'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']
        except:
            state_dict_g = prefix_load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            state_dict_mpd, state_dict_mrd, steps, last_epoch = prefix_load_checkpoint_discriminator(cp_do, device)
            raf_generator.load_state_dict(state_dict_g)
            steps = steps + 1
            raf_mpd.load_state_dict(state_dict_mpd)
            raf_mrd.load_state_dict(state_dict_mrd)

    # Load optimizer states
    if state_dict_do is not None:
        print("Loading RAF Optimizer States...")
        try:
            raf_optimizer_g.load_state_dict(state_dict_do['optim_g'])
        except:
            raf_optimizer_g.load_state_dict(state_dict_do["optimizer"])
        raf_optimizer_d.load_state_dict(state_dict_do['optim_d'])

    # Setup distributed data parallel if using multiple GPUs
    if n_gpus > 1:
        raf_generator = DDP(raf_generator, device_ids=[rank])
        raf_mpd = DDP(raf_mpd, device_ids=[rank]).to(device)
        raf_mrd = DDP(raf_mrd, device_ids=[rank]).to(device)

    # Set models to training mode
    raf_generator.train()
    raf_mpd.train()
    raf_mrd.train()
    
    # Initialize learning rate schedulers
    raf_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(raf_optimizer_g, gamma=hps.lr_decay, last_epoch=last_epoch)
    raf_scheduler_d = torch.optim.lr_scheduler.ExponentialLR(raf_optimizer_d, gamma=hps.lr_decay, last_epoch=last_epoch)

    # Main training loop
    for epoch in range(max(0, last_epoch), a.training_epochs):
        # Get current gamma value for gradient penalty
        current_gamma = raf_gamma_scheduler.get_gamma(epoch) 
        start = time.time()
        
        if rank == 0:
            print("Epoch: {:d}".format(epoch))
            print('Learning Rate : {:.6f}'.format(raf_optimizer_g.param_groups[0]['lr']))
            raf_train(a, rank, epoch, hps, raf_generator, [raf_mpd, raf_mrd], [raf_optimizer_g, raf_optimizer_d], 
                     [raf_scheduler_g, raf_scheduler_d], [train_loader, validation_loader], n_gpus, current_gamma)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))
        else:
            raf_train(a, rank, epoch, hps, raf_generator, [raf_mpd, raf_mrd], [raf_optimizer_g, raf_optimizer_d], 
                     [raf_scheduler_g, raf_scheduler_d], [train_loader, None], n_gpus, current_gamma)

def raf_train(a, rank, epoch, hps, nets, discs, optims, schedulers, loaders, n_gpus, current_gamma):
    """
    RAF training function implementing the core RAF methodology:
    1. Quality Gap: Uses SSL models (WavLM, HuBERT) + M-STFT for perceptual quality assessment
    2. Discriminator Gap: Relativistic pairing of real/fake samples using softplus activation
    3. Adversarial Training: Minimizes discrepancy between quality gap and discriminator gap
    """
    
    # Load RAF quality gap estimators (SSL models for perceptual guidance)
    
    # WavLM-Large for SSL-based quality assessment
    checkpoint = torch.load("./WavLM-Large.pt")
    wavlm_cfg = WavLMConfig(checkpoint['cfg'])
    raf_wavlm = WavLM(wavlm_cfg).to('cuda')
    raf_wavlm.load_state_dict(checkpoint['model'])
    raf_wavlm.eval()
    
    # HuBERT-Large for complementary SSL features
    bundle = torchaudio.pipelines.HUBERT_LARGE
    raf_hubert = bundle.get_model().to('cuda')
    raf_hubert.eval()
    
    # Multi-scale STFT loss for frequency-domain quality assessment
    raf_msstft_loss = QMultiScaleSTFTLoss(window_lengths = [4096, 2048, 1024, 512, 256]).to('cuda')
    raf_msstft_loss.eval()
    
    # SCOREQ model for reconstruction loss
    raf_scoreq_model = Scoreq(data_domain='synthetic', mode='ref', device='cuda')

    def compute_ssl_embedding_distance(model, model_name, gt, pred, device):
        """
        Compute SSL embedding distance as part of RAF quality gap.
        Uses normalized embeddings and MSE distance in embedding space.
        """
        gt = gt.to(device)
        pred = pred.to(device)
        
        if model_name == "wavlm":
            # WavLM uses last convolutional layer features
            gt = F.layer_norm(gt, gt.shape).squeeze(1)
            pred = F.layer_norm(pred, pred.shape).squeeze(1)
            with torch.inference_mode():
                gt_rep = model.extract_features(gt, output_layer = 0) # [B, sequence_length, C]
                pred_rep = model.extract_features(pred, output_layer = 0)
            # L2 normalize for unit norm as in RAF paper
            gt_rep = torch.nn.functional.normalize(gt_rep, dim=(1,2))
            pred_rep = torch.nn.functional.normalize(pred_rep, dim=(1,2))
        elif model_name == "hubert":
            # HuBERT uses 22nd layer for phoneme-related features
            gt = gt.squeeze(1)
            pred = pred.squeeze(1)
            with torch.inference_mode():
                gt_feats, _ = model.extract_features(gt)
                pred_feats, _ = model.extract_features(pred)
                del _
                gt_rep = gt_feats[21] # 22nd layer (0-indexed)
                pred_rep = pred_feats[21]
                # L2 normalize embeddings
                gt_rep = torch.nn.functional.normalize(gt_rep, dim=(1,2))
                pred_rep = torch.nn.functional.normalize(pred_rep, dim=(1,2))

        # Compute MSE distance in normalized embedding space
        squared_diff = (gt_rep - pred_rep) ** 2
        squared_diff = torch.mean(squared_diff, dim=(1, 2))  # [B]

        return squared_diff.unsqueeze(1) # [B, 1]  

    def compute_raf_quality_gap(gt, pred, device):
        """
        Compute RAF Quality Gap Q(y, G(x)) using three components:
        1. M-STFT distance for spectral patterns
        2. WavLM distance for perceptual quality (scaled by αW=40000)
        3. HuBERT distance for phonemic content (scaled by αH=90000)
        """
        # M-STFT distance for multiple resolution spectral analysis
        gt_as = AudioSignal(gt.squeeze(1), hps.sampling_rate)
        pred_as = AudioSignal(pred.squeeze(1), hps.sampling_rate)
        msstft_distance = raf_msstft_loss(gt_as, pred_as) * 0.5  # αM = 0.5
        
        # Downsample to 16kHz for SSL models (WavLM and HuBERT operate at 16kHz)
        gt = downsample_speech_cuda(gt, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        pred = downsample_speech_cuda(pred, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        
        # WavLM distance with scaling factor αW
        wavlm_distance = compute_ssl_embedding_distance(model=raf_wavlm, model_name="wavlm", gt=gt, pred=pred, device=device) * 40000
        
        # HuBERT distance with scaling factor αH  
        hubert_distance = compute_ssl_embedding_distance(model=raf_hubert, model_name="hubert", gt=gt, pred=pred, device=device) * 90000

        # Concatenate all quality gap components [αM*QM, αW*QW, αH*QH]
        return torch.cat((msstft_distance, wavlm_distance, hubert_distance), dim = 1)
    
    # Unpack training components
    raf_model = nets
    raf_mpd, raf_mrd = discs
    raf_optimizer_g, raf_optimizer_d = optims
    raf_scheduler_g, raf_scheduler_d = schedulers
    train_loader, eval_loader = loaders

    global steps

    # Set epoch for distributed sampler
    if n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)

    # Set models to training mode
    raf_model.train()
    raf_mpd.train()
    raf_mrd.train()

    # Training loop over batches
    for i, batch in enumerate(train_loader):
        if rank == 0:
            start_b = time.time()
        
        # Unpack batch data
        x, y, _, y_mel = batch
        
        x = torch.autograd.Variable(x.to('cuda', non_blocking=True))
        y = torch.autograd.Variable(y.to('cuda', non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to('cuda', non_blocking=True))
        y = y.unsqueeze(1) #[B, 1, sequence_length]

        # Generate fake waveform G(x)
        y_g_hat = raf_model(x) #[B, 1, sequence_length]

        # Generate mel spectrogram for reconstruction loss
        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hps.n_fft, hps.num_mels, hps.sampling_rate, hps.hop_size, hps.win_size,
                                        hps.fmin, hps.fmax_for_loss)
        
        # ===== RAF DISCRIMINATOR TRAINING =====
        raf_optimizer_d.zero_grad()

        # Apply zero-centered gradient penalty every 3 steps (RAF regularization strategy)
        if steps % 3 != 0: # Standard RAF discriminator loss without gradient penalty
   
            # Compute RAF Quality Gap Q(y, G(x))
            raf_quality_gap = compute_raf_quality_gap(gt=y, pred=y_g_hat.detach(), device='cuda').detach()

            # RAF MPD discriminator gap computation
            y_df_hat_r, y_df_hat_g, _, _ = raf_mpd(y, y_g_hat.detach())

            # Compute RAF Discriminator Gap d(y, G(x)) = f(D(y) - D(G(x))) using softplus
            raf_discriminator_gap_mpd = discriminator_gap(y_df_hat_r, y_df_hat_g, big="dr").to('cuda')
            n_mpd = raf_discriminator_gap_mpd.shape[0]

            # Repeat quality gap for each discriminator output
            raf_quality_gap_mpd = repeat_quality_gap(raf_quality_gap, n_mpd).to('cuda')  # [N, B, 3]
            
            # RAF MPD Loss: minimize ||d(y, G(x)) - Q(y, G(x))||²
            raf_mpd_loss = F.mse_loss(raf_discriminator_gap_mpd, raf_quality_gap_mpd)
            raf_quality_gap_vis = torch.mean(torch.mean(raf_quality_gap_mpd.detach(), dim=0), dim=0).cpu().numpy()

            # RAF MRD discriminator gap computation
            y_ds_hat_r, y_ds_hat_g, _, _ = raf_mrd(y, y_g_hat.detach())

            # Compute RAF Discriminator Gap for MRD
            raf_discriminator_gap_mrd = discriminator_gap(y_ds_hat_r, y_ds_hat_g, big="dr").to('cuda')  # [N, B, 3]
            
            n_mrd = raf_discriminator_gap_mrd.shape[0]  # number of discriminators
            raf_quality_gap_mrd = repeat_quality_gap(raf_quality_gap, n_mrd).to('cuda')  # [N, B, 3]
            
            # RAF MRD Loss: minimize ||d(y, G(x)) - Q(y, G(x))||²
            raf_mrd_loss = F.mse_loss(raf_discriminator_gap_mrd, raf_quality_gap_mrd)
            
            # Total RAF discriminator loss
            raf_total_disc_loss = raf_mpd_loss + raf_mrd_loss
            raf_total_disc_loss.backward()
            raf_optimizer_d.step()

        else: # Apply RAF zero-centered gradient penalty (R1 + R2 regularization)
   
            # Compute RAF Quality Gap
            raf_quality_gap = compute_raf_quality_gap(gt=y, pred=y_g_hat.detach(), device='cuda').detach()

            # RAF MPD with gradient penalty
            y_df_hat_r, y_df_hat_g, _, _ = raf_mpd(y, y_g_hat.detach())

            # RAF Discriminator Gap computation
            raf_discriminator_gap_mpd = discriminator_gap(y_df_hat_r, y_df_hat_g, big="dr").to('cuda')
            n_mpd = raf_discriminator_gap_mpd.shape[0]

            raf_quality_gap_mpd = repeat_quality_gap(raf_quality_gap, n_mpd).to('cuda')  # [N, B, 3]

            raf_mpd_loss = F.mse_loss(raf_discriminator_gap_mpd, raf_quality_gap_mpd)
            
            # RAF Zero-centered gradient penalty for MPD (R1 + R2)
            raf_gradient_penalty_mpd_r1, raf_gradient_penalty_mpd_r2 = zero_centered_gradient_penalty(raf_mpd, y, y_g_hat.detach(), device='cuda')
            raf_mpd_gp = raf_gradient_penalty_mpd_r1 + raf_gradient_penalty_mpd_r2

            # For debugging and visualization
            raf_quality_gap_vis = torch.mean(torch.mean(raf_quality_gap_mpd.detach(), dim=0), dim=0).cpu().numpy()

            # RAF MRD with gradient penalty
            y_ds_hat_r, y_ds_hat_g, _, _ = raf_mrd(y, y_g_hat.detach())

            # RAF Discriminator Gap for MRD
            raf_discriminator_gap_mrd = discriminator_gap(y_ds_hat_r, y_ds_hat_g, big="dr").to('cuda')  # [N, B, 3]
            n_mrd = raf_discriminator_gap_mrd.shape[0]
            raf_quality_gap_mrd = repeat_quality_gap(raf_quality_gap, n_mrd).to('cuda')  # [N, B, 3]
            
            raf_mrd_loss = F.mse_loss(raf_discriminator_gap_mrd, raf_quality_gap_mrd)
            
            # RAF Zero-centered gradient penalty for MRD
            raf_gradient_penalty_mrd_r1, raf_gradient_penalty_mrd_r2 = zero_centered_gradient_penalty(raf_mrd, y, y_g_hat.detach(), device='cuda')
            raf_mrd_gp = raf_gradient_penalty_mrd_r1 + raf_gradient_penalty_mrd_r2

            # Total RAF discriminator loss with gradient penalty
            raf_total_disc_loss = raf_mpd_loss + raf_mrd_loss + current_gamma * 3 * raf_mpd_gp + current_gamma * raf_mrd_gp 
            raf_total_disc_loss.backward()
            raf_optimizer_d.step()
            
        # Clean up discriminator variables
        del y_df_hat_r, y_df_hat_g, raf_discriminator_gap_mpd, y_ds_hat_r, y_ds_hat_g, raf_discriminator_gap_mrd

        # ===== RAF GENERATOR TRAINING =====
        raf_optimizer_g.zero_grad()
        
        # RAF Mel-spectrogram reconstruction loss
        raf_mel_loss = F.l1_loss(y_mel, y_g_hat_mel) 

        # RAF SCOREQ reconstruction loss (operates on 16kHz)
        gt_16k = downsample_speech_cuda(y, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        pred_16k = downsample_speech_cuda(y_g_hat, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        raf_scoreq_recon_loss = torch.mean(raf_scoreq_model.predict(test_path = pred_16k, ref_path = gt_16k))
        
        # RAF Generator discriminator outputs
        gy_df_hat_r, gy_df_hat_g, fmap_f_r, fmap_f_g = raf_mpd(y, y_g_hat)
        gy_ds_hat_r, gy_ds_hat_g, fmap_s_r, fmap_s_g = raf_mrd(y, y_g_hat)

        # RAF Generator Loss: minimize discriminator gap d(y, G(x))
        raf_gen_discriminator_gap_mpd = discriminator_gap(gy_df_hat_r, gy_df_hat_g, big="dr")
        raf_gen_discriminator_gap_mrd = discriminator_gap(gy_ds_hat_r, gy_ds_hat_g, big="dr")   

        raf_gen_mpd_loss = torch.mean(raf_gen_discriminator_gap_mpd)
        raf_gen_mrd_loss = torch.mean(raf_gen_discriminator_gap_mrd)
        
        # RAF Feature matching losses
        raf_feature_loss_mpd = feature_loss(fmap_f_r, fmap_f_g)
        raf_feature_loss_mrd = feature_loss(fmap_s_r, fmap_s_g)
        
        # RAF Total generator loss with weighted components
        raf_total_gen_loss = (raf_gen_mpd_loss + raf_gen_mrd_loss + 
                             raf_feature_loss_mrd + raf_feature_loss_mpd + 
                             raf_mel_loss * 26 + raf_scoreq_recon_loss * 0.1)
        raf_total_gen_loss.backward()
        raf_optimizer_g.step()
        
        # Clean up generator variables
        del pred_16k, gt_16k

        # ===== RAF LOGGING AND CHECKPOINTING =====
        if rank == 0:
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                print(
                    'Steps : {:d}, RAF Gen Loss Total : {:.3f}, Mel-Spec. Error : {:.3f}, RAF MPD Loss : {:.3f}, RAF MRD Loss : {:.3f}, RAF MPD Gen : {:.3f}, RAF MRD Gen : {:.3f}, RAF GP MPD : {:.3f}, RAF GP MRD : {:.3f}, s/b : {:4.3f}'.
                    format(steps, raf_total_gen_loss, mel_error, raf_mpd_loss, raf_mrd_loss, raf_gen_mpd_loss, raf_gen_mrd_loss, raf_mpd_gp,
                            raf_mrd_gp, time.time() - start_b))

            # RAF checkpointing
            if steps % (a.checkpoint_interval) == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': raf_model.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'mpd': raf_mpd.state_dict(),
                                    'mrd': raf_mrd.state_dict(),
                                    'optimizer': raf_optimizer_g.state_dict(), 'optim_d': raf_optimizer_d.state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})

            # RAF Tensorboard/Wandb logging
            if steps % a.summary_interval == 0:
                wandb.log({"raf_generator/gen_loss_total": raf_total_gen_loss, "steps": steps}) 
                wandb.log({"raf_generator/mel_spec_error": mel_error, "steps": steps})
                wandb.log({"raf_generator/mpd_feature_loss": raf_feature_loss_mpd, "steps": steps})
                wandb.log({"raf_generator/mrd_feature_loss": raf_feature_loss_mrd, "steps": steps})
                wandb.log({"raf_discriminator/mpd_error": raf_mpd_loss, "steps": steps})
                wandb.log({"raf_discriminator/mrd_error": raf_mrd_loss, "steps": steps})
                wandb.log({"raf_generator/generator_mpd_error": raf_gen_mpd_loss, "steps": steps})
                wandb.log({"raf_generator/generator_mrd_error": raf_gen_mrd_loss, "steps": steps})
                wandb.log({"raf_discriminator/gradient_penalty_mpd": raf_mpd_gp, "steps": steps})
                wandb.log({"raf_discriminator/gradient_penalty_mrd": raf_mrd_gp, "steps": steps})
                wandb.log({"training_epochs": epoch, "steps": steps})
                wandb.log({"learning_rate": raf_optimizer_g.param_groups[0]['lr'], "steps": steps})
                wandb.log({"raf_quality/msstft": raf_quality_gap_vis[0], "steps": steps})
                wandb.log({"raf_quality/wavlm": raf_quality_gap_vis[1], "steps": steps})
                wandb.log({"raf_quality/hubert": raf_quality_gap_vis[2], "steps": steps})
                wandb.log({"raf_quality/scoreq_recon_loss": raf_scoreq_recon_loss, "steps": steps})
                wandb.log({"raf_gamma": current_gamma, "steps": steps})

            # RAF Validation
            if steps % a.validation_interval == 0 and steps != 0:
                raf_model.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(eval_loader):
                        x, y, _, y_mel = batch
                        y_g_hat = raf_model(x.to('cuda'))
                        y_mel = torch.autograd.Variable(y_mel.to('cuda', non_blocking=True))

                        # Log mel spectrogram error
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hps.n_fft, hps.num_mels, hps.sampling_rate,
                                                        hps.hop_size, hps.win_size,
                                                        hps.fmin, hps.fmax_for_loss)
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                    wandb.log({"raf_validation/validation_mel_spec_error": val_err_tot, "steps": steps})
        steps += 1

    # Update RAF learning rate schedulers
    raf_scheduler_g.step()
    raf_scheduler_d.step()

def main():
    """
    Main function to initialize RAF training with BigVGAN-base architecture.
    RAF achieves superior performance with only 12% of BigVGAN parameters.
    """
    print('Initializing RAF Training Process..')

    # RAF training configuration
    a = easydict.EasyDict({
    "group_name" : None,
    "input_wavs_dir": "./LibriTTS/",
    "input_mels_dir": 'ft_dataset',
    "input_training_file": './train-full.txt',
    "input_validation_file": './val-full.txt',
    "checkpoint_path": './bigvganbase_raf',
    "config": './bigvgan_base_100band_24khz.json',  # RAF uses longer segments for quality gap estimation
    "training_epochs": 45,
    "stdout_interval": 2500,
    "checkpoint_interval": 50000,
    "summary_interval": 2500,
    "validation_interval": 50000,
    "fine_tuning": False,
    "experiment_name": "bigvganbase_raf"
    })

    # Ensure CUDA availability for RAF training
    assert torch.cuda.is_available(), "RAF training requires CUDA."

    # Load RAF configuration
    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    hps = AttrDict(json_config)

    # Setup multi-GPU RAF training
    n_gpus = torch.cuda.device_count()
    hps.batch_size = hps.batch_size // n_gpus  # Divide batch size by number of GPUs
    print("RAF Batch size per GPU is set to:", hps.batch_size)

    # Build RAF training environment
    build_env(a.config, 'config_v1.json', a.checkpoint_path)
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps.num_gpus = n_gpus
    
    # Launch RAF training
    if n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, a, hps,))
    else:
        run(0, n_gpus, a, hps)

if __name__ == "__main__":
    main()
