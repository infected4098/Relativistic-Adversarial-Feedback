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
from model.daf_discriminator import MultiPeriodDiscriminator, MultiResolutionDiscriminator, \
    feature_loss, generator_loss, discriminator_loss
from quality_model.scoreq import Scoreq
from utils import downsample_speech_cuda, zero_centered_gradient_penalty, dydg_asym, repeat_qydiffqg, prefix_load_checkpoint, prefix_load_checkpoint_discriminator
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

class GammaScheduler():
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

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def run(rank, n_gpus, a, hps):
    
    start_gamma = 0.1
    end_gamma = 0.08
    gamma_schedule = GammaScheduler(start_gamma, end_gamma, a.training_epochs)

    global steps

    if  n_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    
    torch.manual_seed(hps.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(
        training_filelist,
        hps,
        hps.segment_size,
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
        #print("Batch size per gpu is :", int(hps.batch_size))

        # Wandb
        wandb.init(project=f"{a.experiment_name}", resume="allow")
        wandb.run.name = ""
        wandb.config.update(hps)


    generator = BigVGAN(hps).cuda()

    mpd = MultiPeriodDiscriminator(hps).cuda()

    if hps.get("use_mbd_instead_of_mrd", False):  # Switch to MBD
        print(
            "[INFO] using MultiBandDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        # Variable name is kept as "mrd" for backward compatibility & minimal code change
        mrd = MultiBandDiscriminator(hps).cuda()
    elif hps.get("use_cqtd_instead_of_mrd", False):  # Switch to CQTD
        print(
            "[INFO] using MultiScaleSubbandCQTDiscriminator of BigVGAN-v2 instead of MultiResolutionDiscriminator"
        )
        mrd = MultiScaleSubbandCQTDiscriminator(hps).cuda()
    else:  # Fallback to original MRD in BigVGAN-v1
        mrd = MultiResolutionDiscriminator(hps, device).cuda()


    if rank == 0:
        num_param = get_param_num(generator)
        print('number of Parameters for Generator:', num_param)
        print("number of Parameters for MPD:  ", get_param_num(mpd))
        print("number of Parameters for MRD:  ", get_param_num(mrd))

        #utmos_model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).cuda()

    else:
        utmos_model = None
        scoreq_val = None


    optimizer = torch.optim.AdamW(generator.parameters(), hps.learning_rate, betas=[hps.adam_b1, hps.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                                hps.learning_rate, betas=[hps.adam_b1, hps.adam_b2])



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
            generator.load_state_dict(state_dict_g['generator'])
            mpd.load_state_dict(state_dict_do['mpd'])
            mrd.load_state_dict(state_dict_do['mrd'])
            steps = state_dict_do['steps'] + 1
            last_epoch = state_dict_do['epoch']
        except:
            state_dict_g = prefix_load_checkpoint(cp_g, device)
            state_dict_do = load_checkpoint(cp_do, device)
            state_dict_mpd, state_dict_mrd, steps, last_epoch = prefix_load_checkpoint_discriminator(cp_do, device)
            generator.load_state_dict(state_dict_g)
            steps = steps + 1
            mpd.load_state_dict(state_dict_mpd)
            mrd.load_state_dict(state_dict_mrd)

    if state_dict_do is not None:
        print("Optimizer Loading...")
        try:
            optimizer.load_state_dict(state_dict_do['optim_g'])
        except:
            optimizer.load_state_dict(state_dict_do["optimizer"])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    if  n_gpus > 1:
        generator = DDP(generator, device_ids=[rank])
        mpd = DDP(mpd, device_ids=[rank]).to(device)
        mrd = DDP(mrd, device_ids=[rank]).to(device)

    generator.train()
    mpd.train()
    mrd.train()
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hps.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.lr_decay, last_epoch=last_epoch)


    for epoch in range(max(0, last_epoch), a.training_epochs):
        current_gamma = gamma_schedule.get_gamma(epoch) 
        start = time.time()
        if rank == 0:
            print("Epoch: {:d}".format(epoch))
            print('Learning Rate : {:.6f}'.format(optimizer.param_groups[0]['lr']))
            train(a, rank, epoch, hps, generator, [mpd,mrd], [optimizer,optim_d], [scheduler_g, scheduler_d],
                               [train_loader, validation_loader], n_gpus, current_gamma)
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

        else:
            train(a, rank, epoch, hps, generator, [mpd,mrd], [optimizer,optim_d], [scheduler_g, scheduler_d],
                               [train_loader, None], n_gpus, current_gamma)

def train(a, rank, epoch, hps, nets, discs, optims, schedulers, loaders, n_gpus, current_gamma):

    # WavLM
    checkpoint = torch.load("./WavLM-Large.pt")
    wavlm_cfg = WavLMConfig(checkpoint['cfg'])
    wavlm = WavLM(wavlm_cfg).to('cuda')
    wavlm.load_state_dict(checkpoint['model'])
    wavlm.eval()
    # HuBERT
    bundle = torchaudio.pipelines.HUBERT_LARGE
    hubert = bundle.get_model().to('cuda')
    hubert.eval()
    # Multi-scale STFT loss
    msstftloss = QMultiScaleSTFTLoss(window_lengths = [4096, 2048, 1024, 512, 256]).to('cuda')
    #msstftloss = QMultiScaleSTFTLoss(window_lengths = [2048, 1024, 512]).to(device)
    msstftloss.eval()
    scoreq_model_recon = Scoreq(data_domain='synthetic', mode='ref', device='cuda')


    def embed_loss(model, model_name, gt, pred, device):
        gt = gt.to(device)
        pred = pred.to(device)
        if model_name == "wavlm":
            gt = F.layer_norm(gt, gt.shape).squeeze(1)
            pred = F.layer_norm(pred, pred.shape).squeeze(1)
            with torch.inference_mode():
                gt_rep = model.extract_features(gt, output_layer = 0) # [B, sequence_length, C]
                pred_rep = model.extract_features(pred, output_layer = 0)
            gt_rep = torch.nn.functional.normalize(gt_rep, dim=(1,2))
            pred_rep = torch.nn.functional.normalize(pred_rep, dim=(1,2))
        elif model_name == "hubert":
            gt = gt.squeeze(1)
            pred = pred.squeeze(1)
            with torch.inference_mode():
                gt_feats, _ = model.extract_features(gt)
                pred_feats, _ = model.extract_features(pred)
                del _
                gt_rep = gt_feats[21] # [B, sequence_length, C]
                pred_rep = pred_feats[21]
                gt_rep = torch.nn.functional.normalize(gt_rep, dim=(1,2))
                pred_rep = torch.nn.functional.normalize(pred_rep, dim=(1,2))

        squared_diff = (gt_rep - pred_rep) ** 2
        squared_diff = torch.mean(squared_diff, dim=(1, 2))  # [B]

        return squared_diff.unsqueeze(1) # [B, 1]  
    def qyqg(gt, pred, device):
        gt_as = AudioSignal(gt.squeeze(1), hps.sampling_rate)
        pred_as = AudioSignal(pred.squeeze(1), hps.sampling_rate)
        msstft_distance = msstftloss(gt_as, pred_as) * 0.5
        gt = downsample_speech_cuda(gt, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        pred = downsample_speech_cuda(pred, hps.sampling_rate, 16000) #[B, 1, sequence_length] 
        wavlm_distance = embed_loss(model=wavlm, model_name="wavlm", gt=gt, pred=pred, device=device) * 40000  # [B, 1]. 15 when the transformer features
        hubert_distance = embed_loss(model=hubert, model_name="hubert", gt=gt, pred=pred, device=device) * 90000

        return torch.cat((msstft_distance, wavlm_distance, hubert_distance), dim = 1)
    
    model = nets
    mpd, mrd = discs
    optimizer, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders

    global steps

    if  n_gpus > 1:
        train_loader.sampler.set_epoch(epoch)

    model.train()
    mpd.train()
    mrd.train()

    for i, batch in enumerate(train_loader):
        if rank == 0:
            start_b = time.time()
        x, y, _, y_mel = batch
        
        x = torch.autograd.Variable(x.to('cuda', non_blocking=True))
        y = torch.autograd.Variable(y.to('cuda', non_blocking=True))
        y_mel = torch.autograd.Variable(y_mel.to('cuda', non_blocking=True))
        y = y.unsqueeze(1) #[B, 1, sequence_length]

        y_g_hat = model(x) #[B, 1, sequence_length]

        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hps.n_fft, hps.num_mels, hps.sampling_rate, hps.hop_size, hps.win_size,
                                        hps.fmin, hps.fmax_for_loss)
        
        optim_d.zero_grad()

        if steps % 3 != 0: # No gradient penalty
   
            qy_qg = qyqg(gt=y, pred=y_g_hat.detach(), device='cuda').detach()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            dy_dg_mpd = dydg_asym(y_df_hat_r, y_df_hat_g, big="dr").to('cuda')
            n_mpd = dy_dg_mpd.shape[0]

            qy_qg_mpd = repeat_qydiffqg(qy_qg, n_mpd).to('cuda')  # [N, B, 3]
            #mpd_loss = F.mse_loss(dy_dg_mpd, qy_qg_mpd) + torch.mean(dy_dg_mpd_rp) # [1]
            mpd_loss = F.mse_loss(dy_dg_mpd, qy_qg_mpd)
            qy_qg_vis = torch.mean(torch.mean(qy_qg_mpd.detach(), dim=0), dim=0).cpu().numpy()

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            dy_dg_msd = dydg_asym(y_ds_hat_r, y_ds_hat_g, big="dr").to('cuda')  # [N, B, 3]
            
            n_msd = dy_dg_msd.shape[0]  # the number of discriminators
            qy_qg_msd = repeat_qydiffqg(qy_qg, n_msd).to('cuda')  # [N, B, 3]. N identical tensors.
            #msd_loss = F.mse_loss(dy_dg_msd, qy_qg_msd) + torch.mean(dy_dg_msd_rp)  # [1]
            msd_loss = F.mse_loss(dy_dg_msd, qy_qg_msd)
            loss_disc_all = mpd_loss + msd_loss
            loss_disc_all.backward()
            #grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000)
            #grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000)
            optim_d.step()

        else: # No gradient penalty
   
            qy_qg = qyqg(gt=y, pred=y_g_hat.detach(), device='cuda').detach()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            dy_dg_mpd = dydg_asym(y_df_hat_r, y_df_hat_g, big="dr").to('cuda')
            n_mpd = dy_dg_mpd.shape[0]

            qy_qg_mpd = repeat_qydiffqg(qy_qg, n_mpd).to('cuda')  # [N, B, 3]
            #mpd_loss = F.mse_loss(dy_dg_mpd, qy_qg_mpd) + torch.mean(dy_dg_mpd_rp) # [1]

            mpd_loss = F.mse_loss(dy_dg_mpd, qy_qg_mpd)
            gradient_penalty_mpd_r1, gradient_penalty_mpd_r2 = zero_centered_gradient_penalty(mpd, y, y_g_hat.detach(), device='cuda')
            mpd_gp = gradient_penalty_mpd_r1 + gradient_penalty_mpd_r2
            qy_qg_vis = torch.mean(torch.mean(qy_qg_mpd.detach(), dim=0), dim=0).cpu().numpy()

            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            dy_dg_msd = dydg_asym(y_ds_hat_r, y_ds_hat_g, big="dr").to('cuda')  # [N, B, 3]
            n_msd = dy_dg_msd.shape[0]  # the number of discriminators
            qy_qg_msd = repeat_qydiffqg(qy_qg, n_msd).to('cuda')  # [N, B, 3]. N identical tensors.
            #msd_loss = F.mse_loss(dy_dg_msd, qy_qg_msd) + torch.mean(dy_dg_msd_rp)  # [1]
            msd_loss = F.mse_loss(dy_dg_msd, qy_qg_msd)
            gradient_penalty_msd_r1, gradient_penalty_msd_r2 = zero_centered_gradient_penalty(mrd, y, y_g_hat.detach(), device='cuda')
            msd_gp = gradient_penalty_msd_r1 + gradient_penalty_msd_r2


            loss_disc_all = mpd_loss + msd_loss + current_gamma * 3 * mpd_gp  + current_gamma * msd_gp 
            loss_disc_all.backward()
            #grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000)
            #grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000)
            optim_d.step()
        del y_df_hat_r, y_df_hat_g, dy_dg_mpd, y_ds_hat_r, y_ds_hat_g, dy_dg_msd

        # Generator
        # MPD loss
        optimizer.zero_grad()
        loss_mel = F.l1_loss(y_mel, y_g_hat_mel) 
        gt_16k = downsample_speech_cuda(y, hps.sampling_rate, 16000) #[B, 1, sequence_length] # Only when 22k 
        pred_16k = downsample_speech_cuda(y_g_hat, hps.sampling_rate, 16000) #[B, 1, sequence_length] # Only when 22k 
        scoreq_recon_loss = torch.mean(scoreq_model_recon.predict(test_path = pred_16k, ref_path = gt_16k))
        gy_df_hat_r, gy_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
        gy_ds_hat_r, gy_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)

        # Generator loss

        #dy_dg_gen_mpd = dydg_asym(gy_df_hat_r, gy_df_hat_g, big="dg")
        #dy_dg_gen_msd = dydg_asym(gy_ds_hat_r, gy_ds_hat_g, big="dg")
        dy_dg_gen_mpd_rp = dydg_asym(gy_df_hat_r, gy_df_hat_g, big="dr")
        dy_dg_gen_msd_rp = dydg_asym(gy_ds_hat_r, gy_ds_hat_g, big="dr")   

        loss_gen_mpd = torch.mean(dy_dg_gen_mpd_rp)
        loss_gen_msd = torch.mean(dy_dg_gen_msd_rp)
        # Feature matching loss
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_all = loss_gen_mpd + loss_gen_msd + loss_fm_s + loss_fm_f + loss_mel * 26 + scoreq_recon_loss * 0.1
        loss_gen_all.backward()
        optimizer.step()
        del pred_16k, gt_16k

        if rank == 0:
            if steps % a.stdout_interval == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                print(
                    'Steps : {:d}, Gen Loss Total : {:.3f}, Mel-Spec. Error : {:.3f}, MPDLoss : {:.3f}, MRDLoss : {:.3f}, MPD_gen : {:.3f}, MRD_gen : {:.3f}, GP_mpd : {:.3f}, GP_mrd : {:.3f}, s/b : {:4.3f}'.
                    format(steps, loss_gen_all, mel_error, mpd_loss, msd_loss, loss_gen_mpd, loss_gen_msd, mpd_gp,
                            msd_gp, time.time() - start_b))


            # checkpointing
            if steps % (a.checkpoint_interval) == 0 and steps != 0:
                checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'generator': model.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                save_checkpoint(checkpoint_path,
                                {'mpd': mpd.state_dict(),
                                    'mrd': mrd.state_dict(),
                                    'optimizer': optimizer.state_dict(), 'optim_d': optim_d.state_dict(),
                                    'steps': steps,
                                    'epoch': epoch})

            # Tensorboard summary logging
            if steps % a.summary_interval == 0:
                wandb.log({"generator/gen_loss_total": loss_gen_all, "steps": steps}) 
                wandb.log({"generator/mel_spec_error": mel_error, "steps": steps})
                wandb.log({"generator/mpd_feature_loss": loss_fm_f, "steps": steps})
                wandb.log({"generator/mrd_feature_loss": loss_fm_s, "steps": steps})
                wandb.log({"discriminator/mpd_error": mpd_loss, "steps": steps})
                wandb.log({"discriminator/mrd_error": msd_loss, "steps": steps})
                wandb.log({"generator/generator_mpd_error": loss_gen_mpd, "steps": steps})
                wandb.log({"generator/generator_mrd_error": loss_gen_msd, "steps": steps})
                wandb.log({"discriminator/gradient_penalty_mpd": mpd_gp, "steps": steps})
                wandb.log({"discriminator/gradient_penalty_mrd": msd_gp, "steps": steps})
                #wandb.log({"discriminator/rploss_mrd": torch.mean(dy_dg_msd_rp), "steps":steps})
                #wandb.log({"discriminator/rploss_mpd": torch.mean(dy_dg_mpd_rp), "steps":steps})
                wandb.log({"training_epochs": epoch, "steps": steps})
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr'], "steps": steps})
                wandb.log({"quality/msstft": qy_qg_vis[0], "steps": steps})
                wandb.log({"quality/wavlm": qy_qg_vis[1], "steps": steps})
                wandb.log({"quality/hubert": qy_qg_vis[2], "steps": steps})
                wandb.log({"quality/scoreq_recon_loss": scoreq_recon_loss, "steps": steps})
                wandb.log({"gamma": current_gamma, "steps": steps})



            # Validation
            if steps % a.validation_interval == 0 and steps != 0:
            #if steps % a.validation_interval == 0:
                model.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0

                with torch.no_grad():
                    for j, batch in enumerate(eval_loader):
                        x, y, _, y_mel = batch
                        y_g_hat = model(x.to('cuda'))
                        y_mel = torch.autograd.Variable(y_mel.to('cuda', non_blocking=True))
                        y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), hps.n_fft, hps.num_mels, hps.sampling_rate,
                                                        hps.hop_size, hps.win_size,
                                                        hps.fmin, hps.fmax_for_loss)
                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()
                    wandb.log({"validation/validation_mel_spec_error": val_err_tot, "steps": steps})
        steps += 1

    scheduler_g.step()
    scheduler_d.step()


def main():

    print('Initializing Training Process..')

    a = easydict.EasyDict({
    "group_name" : None,
    "input_wavs_dir": "./LibriTTS/",
    "input_mels_dir": 'ft_dataset',
    "input_training_file": './train-full.txt',
    "input_validation_file": './val-full.txt',
    "checkpoint_path": './bigvganbase_raf',
    "config": './bigvgan_base_100band_24khz.json',
    "training_epochs": 45,
    "stdout_interval": 2500,
    "checkpoint_interval": 50000,
    "summary_interval": 2500,
    "validation_interval": 50000,
    "fine_tuning": False,
    "experiment_name": "bigvganbase_raf"
    })

    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."


    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    hps = AttrDict(json_config)

    n_gpus = torch.cuda.device_count()
    hps.batch_size = hps.batch_size // n_gpus  # Divide batch size by number of GPUs
    print("Batch size per GPU is set to:", hps.batch_size)


    build_env(a.config, 'config_v1.json', a.checkpoint_path)
    port = 50000 + random.randint(0, 100)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    hps.num_gpus = n_gpus
    if  n_gpus > 1:
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, a, hps,))
    else:
        run(0, n_gpus, hps)





if __name__ == "__main__":

    main()
