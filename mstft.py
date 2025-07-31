
import torch
import torch.nn.functional as F
from audiotools import AudioSignal
from audiotools import STFTParams
from torch import nn
import typing
from typing import List

class MultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        loss = 0.0
        for i, s in enumerate(self.stft_params):
            x.stft(s.window_length, s.hop_length, s.window_type)
            y.stft(s.window_length, s.hop_length, s.window_type)
            loss += self.log_weight * self.loss_fn(
                x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
            )
            loss += self.mag_weight * self.loss_fn(x.magnitude, y.magnitude)
        return loss / (i + 1)
    

class QMultiScaleSTFTLoss(nn.Module):
    """Computes the multi-scale STFT loss from [1].

    Parameters
    ----------
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [2048, 512]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 1.0
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 2.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    References
    ----------

    1.  Engel, Jesse, Chenjie Gu, and Adam Roberts.
        "DDSP: Differentiable Digital Signal Processing."
        International Conference on Learning Representations. 2019.

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    """

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(reduction="none"),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x: AudioSignal, y: AudioSignal):
        """Computes multi-scale STFT between an estimate and a reference
        signal.

        Parameters
        ----------
        x : AudioSignal
            Estimate signal
        y : AudioSignal
            Reference signal

        Returns
        -------
        torch.Tensor
            Multi-scale STFT loss.
        """
        for i, s in enumerate(self.stft_params):
            if i == 0:
                x.stft(s.window_length, s.hop_length, s.window_type)
                y.stft(s.window_length, s.hop_length, s.window_type)
                loss = torch.mean(self.log_weight * self.loss_fn(
                    x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                    y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                ), dim=(2,3)).T
                loss += torch.mean(self.mag_weight * self.loss_fn(x.magnitude, y.magnitude), dim=(2,3)).T


            else:
                x.stft(s.window_length, s.hop_length, s.window_type)
                y.stft(s.window_length, s.hop_length, s.window_type)
                loss += torch.mean(self.log_weight * self.loss_fn(
                    x.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                    y.magnitude.clamp(self.clamp_eps).pow(self.pow).log10(),
                ), dim=(2,3)).T
                loss += torch.mean(self.mag_weight * self.loss_fn(x.magnitude, y.magnitude), dim=(2,3)).T
        return loss / (i + 1)
    

class QMultiScaleSTFTLoss_torch(nn.Module):

    def __init__(
        self,
        window_lengths: List[int] = [2048, 512],
        loss_fn: typing.Callable = nn.L1Loss(reduction="none"),
        clamp_eps: float = 1e-5,
        mag_weight: float = 1.0,
        log_weight: float = 1.0,
        pow: float = 2.0,
        weight: float = 1.0,
        match_stride: bool = False,
        window_type: str = None,
    ):
        super().__init__()
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.loss_fn = loss_fn
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.clamp_eps = clamp_eps
        self.weight = weight
        self.pow = pow

    def forward(self, x, y):

    
        for i, s in enumerate(self.stft_params):
            if i == 0:
                x_stft = torch.stft(
                    x,
                    n_fft=s.window_length,
                    hop_length=s.hop_length,
                    window=torch.hann_window(s.window_length).to(x.device),
                    return_complex=True
                )

                y_stft = torch.stft(
                    y,
                    n_fft=s.window_length,
                    hop_length=s.hop_length,
                    window=torch.hann_window(s.window_length).to(y.device),
                    return_complex=True
                )
                x_stft
                y_stft
                loss = torch.mean(self.log_weight * self.loss_fn(
                    torch.abs(x_stft).clamp(self.clamp_eps).pow(self.pow).log10(),
                    torch.abs(y_stft).clamp(self.clamp_eps).pow(self.pow).log10(),
                ), dim=(1,2)).T
                loss += torch.mean(self.mag_weight * self.loss_fn(torch.abs(x_stft), torch.abs(y_stft)), dim=(1,2)).T


            else:
                x_stft = torch.stft(
                    x,
                    n_fft=s.window_length,
                    hop_length=s.hop_length,
                    window=torch.hann_window(s.window_length).to(x.device),
                    return_complex=True
                )
                y_stft = torch.stft(
                    y,
                    n_fft=s.window_length,
                    hop_length=s.hop_length,
                    window=torch.hann_window(s.window_length).to(y.device),
                    return_complex=True
                )
                loss += torch.mean(self.log_weight * self.loss_fn(
                    torch.abs(x_stft).clamp(self.clamp_eps).pow(self.pow).log10(),
                    torch.abs(y_stft).clamp(self.clamp_eps).pow(self.pow).log10(),
                ), dim=(1,2)).T
                loss += torch.mean(self.mag_weight * self.loss_fn(torch.abs(x_stft), torch.abs(y_stft)), dim=(1,2)).T
        return loss / (i + 1)
