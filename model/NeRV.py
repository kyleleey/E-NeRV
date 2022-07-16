import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding


def NeRV_MLP(dim_list, act='relu', bias=True):
    act_fn = ActivationLayer(act)
    fc_list = []
    for i in range(len(dim_list) - 1):
        fc_list += [nn.Linear(dim_list[i], dim_list[i+1], bias=bias), act_fn]
    return nn.Sequential(*fc_list)


class NeRV_CustomConv(nn.Module):
    def __init__(self, **kargs):
        super(NeRV_CustomConv, self).__init__()

        ngf, new_ngf, stride = kargs['ngf'], kargs['new_ngf'], kargs['stride']
        self.conv_type = kargs['conv_type']
        if self.conv_type == 'conv':
            self.conv = nn.Conv2d(ngf, new_ngf * stride * stride, 3, 1, 1, bias=kargs['bias'])
            self.up_scale = nn.PixelShuffle(stride)
        elif self.conv_type == 'deconv':
            self.conv = nn.ConvTranspose2d(ngf, new_ngf, stride, stride)
            self.up_scale = nn.Identity()
        elif self.conv_type == 'bilinear':
            self.conv = nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=True)
            self.up_scale = nn.Conv2d(ngf, new_ngf, 2*stride+1, 1, stride, bias=kargs['bias'])

    def forward(self, x):
        out = self.conv(x)
        return self.up_scale(out)


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()

        self.conv = NeRV_CustomConv(ngf=kargs['ngf'], new_ngf=kargs['new_ngf'], stride=kargs['stride'], bias=kargs['bias'], 
            conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Conv_Up_Block(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        ngf = kargs['ngf']
        new_ngf = kargs['new_ngf']

        if ngf <= new_ngf:
            factor = 4
            self.conv1 = NeRV_CustomConv(ngf=ngf, new_ngf=ngf // factor, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])
            self.conv2 = nn.Conv2d(ngf // factor, new_ngf, 3, 1, 1, bias=kargs['bias'])
        else:
            self.conv1 = nn.Conv2d(ngf, new_ngf, 3, 1, 1, bias=kargs['bias'])
            self.conv2 = NeRV_CustomConv(ngf=new_ngf, new_ngf=new_ngf, stride=kargs['stride'], bias=kargs['bias'], conv_type=kargs['conv_type'])
        self.norm = NormLayer(kargs['norm'], kargs['new_ngf'])
        self.act = ActivationLayer(kargs['act'])

    def forward(self, x):
        return self.act(self.norm(self.conv2(self.conv1(x))))


class NeRV_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pe = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim, stem_num = [int(x) for x in cfg['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]
        mlp_dim_list = [self.pe.embed_length] + [stem_dim] * stem_num + [self.fc_h *self.fc_w *self.fc_dim]
        self.stem = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers = [nn.ModuleList() for _ in range(2)]
        ngf = self.fc_dim
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])

            for j in range(cfg['num_blocks']):
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=1 if j else stride,
                    bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
                ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias']) 
                    # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias']) 
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
                # head_layer = nn.Conv2d(ngf, 3, 3, 1, 1, bias=kargs['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

    def forward(self, data):
        input_id = data['img_id']  # [B]
        input_emb = self.pe(input_id) # [B, L]
        output = self.stem(input_emb)
        output = output.view(output.size(0), self.fc_dim, self.fc_h, self.fc_w)

        out_list = []
        for layer, head_layer in zip(self.layers, self.head_layers):
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list