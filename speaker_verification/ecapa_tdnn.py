# Adapted from https://github.com/ASLP-lab/MeanVC (Apache 2.0)
# Original credit: https://github.com/lawlict/ECAPA-TDNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans


class Res2Conv1dReluBn(nn.Module):
    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1
        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.nums)])

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if i == 0 else sp + spx[i]
            sp = self.bns[i](F.relu(self.convs[i](sp)))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        return torch.cat(out, dim=1)


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        return x * out.unsqueeze(2)


class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = self.shortcut(x) if self.shortcut else x
        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)
        return x + residual


class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att
        in_dim_att = in_dim * 3 if global_context_att else in_dim
        self.linear1 = nn.Conv1d(in_dim_att, attention_channels, kernel_size=1)
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)

    def forward(self, x):
        if self.global_context_att:
            ctx_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            ctx_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, ctx_mean, ctx_std), dim=1)
        else:
            x_in = x
        alpha = torch.softmax(self.linear2(torch.tanh(self.linear1(x_in))), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        std = torch.sqrt(torch.sum(alpha * (x ** 2), dim=2) - mean ** 2 + 1e-9)
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, feat_dim=80, channels=512, emb_dim=192, global_context_att=False,
                 feat_type='fbank', sr=16000, feature_selection="hidden_states",
                 update_extract=False, config_path=None):
        super().__init__()
        self.feat_type = feat_type
        self.feature_selection = feature_selection
        self.update_extract = update_extract
        self.sr = sr

        win_len = int(sr * 0.025)
        hop_len = int(sr * 0.01)

        if feat_type == 'fbank':
            self.update_extract = False
            self.feature_extract = trans.MelSpectrogram(
                sample_rate=sr, n_fft=512, win_length=win_len,
                hop_length=hop_len, f_min=0.0, f_max=sr // 2, pad=0, n_mels=feat_dim)
        elif feat_type == 'mfcc':
            self.update_extract = False
            melkwargs = {'n_fft': 512, 'win_length': win_len, 'hop_length': hop_len,
                         'f_min': 0.0, 'f_max': sr // 2, 'pad': 0}
            self.feature_extract = trans.MFCC(sample_rate=sr, n_mfcc=feat_dim, log_mels=False, melkwargs=melkwargs)
        else:
            torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
            self.feature_extract = torch.hub.load('s3prl/s3prl', feat_type)
            if len(self.feature_extract.model.encoder.layers) == 24:
                for layer_idx in [11, 23]:
                    layer = self.feature_extract.model.encoder.layers[layer_idx]
                    if hasattr(layer.self_attn, "fp32_attention"):
                        layer.self_attn.fp32_attention = False
            self.feat_num = self._get_feat_num()
            self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))

        if feat_type not in ('fbank', 'mfcc'):
            freeze_list = ['final_proj', 'label_embs_concat', 'mask_emb', 'project_q', 'quantizer']
            for name, param in self.feature_extract.named_parameters():
                if any(f in name for f in freeze_list):
                    param.requires_grad = False

        if not self.update_extract:
            for param in self.feature_extract.parameters():
                param.requires_grad = False

        self.instance_norm = nn.InstanceNorm1d(feat_dim)
        self.channels = [channels] * 4 + [1536]
        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], 3, 1, 2, 2, 8, 128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], 3, 1, 3, 3, 8, 128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], 3, 1, 4, 4, 8, 128)
        self.conv = nn.Conv1d(channels * 3, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

    def _get_feat_num(self):
        self.feature_extract.eval()
        wav = [torch.randn(self.sr).to(next(self.feature_extract.parameters()).device)]
        with torch.no_grad():
            features = self.feature_extract(wav)
        select = features[self.feature_selection]
        return len(select) if isinstance(select, (list, tuple)) else 1

    def _get_feat(self, x):
        if self.update_extract:
            x = self.feature_extract([s for s in x])
        else:
            with torch.no_grad():
                if self.feat_type in ('fbank', 'mfcc'):
                    x = self.feature_extract(x) + 1e-6
                else:
                    x = self.feature_extract([s for s in x])

        if self.feat_type == 'fbank':
            x = x.log()

        if self.feat_type not in ('fbank', 'mfcc'):
            x = x[self.feature_selection]
            if isinstance(x, (list, tuple)):
                x = torch.stack(x, dim=0)
            else:
                x = x.unsqueeze(0)
            norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            x = (norm_weights * x).sum(dim=0)
            x = x.transpose(1, 2) + 1e-6

        return self.instance_norm(x)

    def forward(self, x):
        x = self._get_feat(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.relu(self.conv(torch.cat([out2, out3, out4], dim=1)))
        out = self.bn(self.pooling(out))
        return self.linear(out)


def ECAPA_TDNN_SMALL(feat_dim, emb_dim=256, feat_type='fbank', sr=16000,
                     feature_selection="hidden_states", update_extract=False, config_path=None):
    return ECAPA_TDNN(feat_dim=feat_dim, channels=512, emb_dim=emb_dim,
                      feat_type=feat_type, sr=sr, feature_selection=feature_selection,
                      update_extract=update_extract, config_path=config_path)
