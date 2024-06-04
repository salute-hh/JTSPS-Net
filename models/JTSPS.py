import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import numpy as np
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter


class PrechooseSMI(nn.Module):
    def __init__(self, kernel_size, skeleton):
        super().__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size // 2, stride=1, padding=0)
        self.skeleton = skeleton

    def forward(self, x):
        batch_size, num_channels, frame_size, _ = x.size()
        pooled_output = self.avg_pool(x)

        num_selected_channels = round(self.skeleton / 2)
        num_choose_channels = math.ceil(self.skeleton / 2)
        selected_data = x.new_zeros(batch_size, num_selected_channels, frame_size, frame_size)
        pooled_output_numpy = pooled_output.cpu().detach().numpy()
        min_data = np.amin(pooled_output_numpy, axis=(2, 3))
        original_min_data = min_data.tolist()

        min_data.sort()

        period_list_batch = []

        for batch_idx in range(batch_size):
            period_list = []
            frame_count = 0
            for i in range(num_choose_channels, self.skeleton):
                channel_idx = original_min_data[batch_idx].index(min_data[batch_idx][i])
                selected_data[batch_idx][frame_count] = x[batch_idx][channel_idx]
                frame_count += 1
                period_list.append(channel_idx)
            
            if frame_count == 0:
                for i in range(num_choose_channels, self.skeleton):
                    channel_idx = original_min_data[batch_idx].index(min_data[batch_idx][i])
                    selected_data[batch_idx][frame_count] = x[batch_idx][channel_idx]
                    frame_count += 1
                    period_list.append(channel_idx)
            
            current_frame = 0
            while frame_count < num_selected_channels:
                if current_frame < frame_count:
                    selected_data[batch_idx][frame_count] = selected_data[batch_idx][current_frame]
                    current_frame += 1
                frame_count += 1

            period_list.sort()
            period_list_batch.append(period_list)

        return selected_data, period_list_batch





class SimilarityMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, attn_mask=None):
        batch_size, num_channels, m, v = query.size()
        _, _, n, _ = key.size()
        query = query.permute(0, 1, 3, 2)
        key = key.permute(0, 1, 3, 2)

        batch_matrix = []
        for batch_idx in range(batch_size):
            channel_matrices = []
            for channel_idx in range(num_channels):
                matrix = self.compute_similarity(query[batch_idx][channel_idx], key[batch_idx][channel_idx], m, n)
                channel_matrices.append(matrix)
            channel_matrices = torch.stack(channel_matrices)
            batch_matrix.append(channel_matrices)

        batch_matrix = torch.stack(batch_matrix)
        return batch_matrix

    def compute_similarity(self, x, y, m, n):
        x0 = torch.unsqueeze(x[0], 0)
        y0 = torch.unsqueeze(y[0], 0)

        dist = self.compute_distance(x0, y0, m, n)

        for i in range(1, 4):
            xi = torch.unsqueeze(x[i], 0)
            yi = torch.unsqueeze(y[i], 0)
            dist += self.compute_distance(xi, yi, m, n)

        return dist.clamp(min=1e-12).sqrt()

    def compute_distance(self, a, b, m, n):
        a_squared_sum = torch.pow(a, 2).sum(0, keepdim=True).expand(m, n)
        b_squared_sum = torch.pow(b, 2).sum(0, keepdim=True).expand(n, m).t()
        dist = a_squared_sum + b_squared_sum
        dist.addmm_(a.t(), b, beta=1, alpha=-2)
        return dist



def get_1d_dct(i, freq, length):
    result = math.cos(math.pi * freq * (i + 0.5) / length) / math.sqrt(length)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)

def get_dct_weights(width, height, channels, fidx_u=None, fidx_v=None):
    if fidx_u is None:
        fidx_u = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3]
    if fidx_v is None:
        fidx_v = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 2, 5]

    scale_ratio = width // 7
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channels, width, height)
    channel_part = channels // len(fidx_u)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * channel_part: (i + 1) * channel_part, t_x, t_y] = (
                    get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
                )

    return dct_weights

class FcaLayer(nn.Module):
    def __init__(self, channels, reduction, width, height, kernel_size=5):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channels))
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (self.height, self.width))
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2, 3)).unsqueeze(-1).unsqueeze(-1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class TransEncoder(nn.Module):
    '''standard transformer encoder'''

    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers=1, num_frames=64):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, num_frames)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=n_head,
                                                   dim_feedforward=dim_ff,
                                                   dropout=dropout,
                                                   activation='relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


class Prediction(nn.Module):
    ''' predict the density map with densenet '''

    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
            # ,
            # nn.ReLU(True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """
    def __init__(
        self,
        num_channels,
        eps = 1e-5,
        affine = True,
        device = None,
        dtype = None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias

        return out


class JTSPSModel(nn.Module):
    def __init__(self, num_frames, skeleton, scale):
        super(JTSPSModel, self).__init__()
        self.num_frames = num_frames
        self.skeleton = skeleton
        self.scale = scale
        convert_conv = round(self.skeleton / 2)

        self.sims = SimilarityMatrix()

        self.prechoose = PrechooseSMI(kernel_size=self.scale[0], skeleton=self.skeleton)
        self.prechoose1 = PrechooseSMI(kernel_size=self.scale[1], skeleton=self.skeleton)
        self.prechoose2 = PrechooseSMI(kernel_size=self.scale[2], skeleton=self.skeleton)

        self.FcaLayer = FcaLayer(channels=convert_conv, reduction=8, width=self.scale[0], height=self.scale[0], kernel_size=3)
        self.FcaLayer1 = FcaLayer(channels=convert_conv, reduction=8, width=self.scale[1], height=self.scale[1], kernel_size=3)
        self.FcaLayer2 = FcaLayer(channels=convert_conv, reduction=8, width=self.scale[2], height=self.scale[2], kernel_size=3)

        self.upsample = nn.Upsample(size=self.scale[1], mode='nearest')
        self.upsample1 = nn.Upsample(size=self.scale[0], mode='nearest')

        self.conv3x3_upsample = nn.Conv2d(in_channels=convert_conv, out_channels=convert_conv, kernel_size=3, padding=1)
        self.conv3x3_upsample1 = nn.Conv2d(in_channels=convert_conv, out_channels=convert_conv, kernel_size=3, padding=1)
        self.conv3x3 = nn.Conv2d(in_channels=convert_conv, out_channels=32, kernel_size=3, padding=1)

        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1, num_frames=self.num_frames)
        self.FC = Prediction(512, 512, 256, 1)
        self.conf_fc = nn.Sequential(
            nn.Linear(64, 1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        with autocast():
            x_scales = self._process_data(x)
            x_sims = self._calculate_similarity_matrices(x_scales)
            x, x1, x2 = self._coarse_fine_selection(x_sims)
            x = self._multi_scale_fusion(x, x1, x2)
            x = self._impulse_regression(x)
            return x

    def _process_data(self, x):
        x_scale_0 = x[:, :, :self.scale[0], :]
        x_scale_1 = x[:, :, self.scale[0]:self.scale[0] + self.scale[1], :]
        x_scale_2 = x[:, :, self.scale[0] + self.scale[1]:, :]

        x_scale1 = self._create_scale_tensors(x_scale_0)
        x_scale2 = self._create_scale_tensors(x_scale_1)
        x_scale3 = self._create_scale_tensors(x_scale_2)

        return [x_scale1, x_scale2, x_scale3]

    def _create_scale_tensors(self, x_scale):
        x1 = x_scale[:, :, :, 0]
        y1 = x_scale[:, :, :, 1]
        B, C, T = x1.shape

        zerox = x1.new_zeros([B, C, 1])
        zeroy = y1.new_zeros([B, C, 1])
        x2 = torch.diff(x1, prepend=zerox)
        y2 = torch.diff(x1, prepend=zeroy)
        return torch.stack((x1, y1, x2, y2), dim=-1).float()

    def _calculate_similarity_matrices(self, x_scales):
        sim1 = self.sims(x_scales[0], x_scales[0])
        sim2 = self.sims(x_scales[1], x_scales[1])
        sim3 = self.sims(x_scales[2], x_scales[2])

        x_sims1 = F.relu(sim1)
        x_sims2 = F.relu(sim2)
        x_sims3 = F.relu(sim3)

        return [x_sims1, x_sims2, x_sims3]

    def _coarse_fine_selection(self, x_sims):
        x, _ = self.prechoose(x_sims[0])
        x = self.FcaLayer(x)

        x1, _ = self.prechoose1(x_sims[1])
        x1 = self.FcaLayer1(x1)

        x2, _ = self.prechoose2(x_sims[2])
        x2 = self.FcaLayer2(x2)

        return x, x1, x2

    def _multi_scale_fusion(self, x, x1, x2):
        x2 = self.upsample(x2)
        x2 = F.relu(self.conv3x3_upsample(x2))

        x1 = x2 + x1
        x1 = self.upsample1(x1)
        x1 = F.relu(self.conv3x3_upsample1(x1))

        x = x1 + x
        x = F.relu(self.bn2(self.conv3x3(x)))
        x = self.dropout1(x)

        return x

    def _impulse_regression(self, x):
        x = x.permute(0, 2, 3, 1).flatten(start_dim=2)
        x = F.relu(self.input_projection(x))
        x = self.ln1(x)

        x = x.transpose(0, 1)
        x = self.transEncoder(x)
        x = x.transpose(0, 1)

        x = self.FC(x).squeeze(2)
        return x

