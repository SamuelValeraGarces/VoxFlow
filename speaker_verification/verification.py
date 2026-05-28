# Adapted from https://github.com/ASLP-lab/MeanVC (Apache 2.0)

import torch
import numpy as np
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from speaker_verification.ecapa_tdnn import ECAPA_TDNN_SMALL


def init_sv_model(model_name: str, checkpoint: str = None):
    if model_name == 'wavlm_large':
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type='wavlm_large')
    elif model_name == 'wavlm_base_plus':
        model = ECAPA_TDNN_SMALL(feat_dim=768, feat_type='wavlm_base_plus')
    else:
        model = ECAPA_TDNN_SMALL(feat_dim=40, feat_type='fbank')

    if checkpoint is not None and Path(checkpoint).exists():
        state_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=False)

    return model


def get_speaker_embedding(model: torch.nn.Module, wav_path: str,
                           device: torch.device, sample_rate: int = 16000) -> torch.Tensor:
    waveform, sr = torchaudio.load(wav_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    waveform = waveform.mean(0, keepdim=True).to(device)
    with torch.no_grad():
        emb = model(waveform)
    return emb


def get_speaker_embedding_from_tensor(model: torch.nn.Module, waveform: torch.Tensor,
                                       device: torch.device) -> torch.Tensor:
    waveform = waveform.to(device)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    with torch.no_grad():
        emb = model(waveform)
    return emb
