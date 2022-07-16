import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as dist
from .model_utils import ActivationLayer, NormLayer, PositionalEncoding, gradient
from .NeRV import NeRV_MLP, NeRVBlock, Conv_Up_Block
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = heads * dim_head
        project_out = not(heads==1 and dim_head==dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0., prenorm=False):
        super(TransformerBlock, self).__init__()
        if prenorm:
            self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
            self.ffn = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        else:
            self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x


class E_NeRV_Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # t mapping
        self.pe_t = PositionalEncoding(
            pe_embed_b=cfg['pos_b'], pe_embed_l=cfg['pos_l']
        )

        stem_dim_list = [int(x) for x in cfg['stem_dim_num'].split('_')]
        self.fc_h, self.fc_w, self.fc_dim = [int(x) for x in cfg['fc_hw_dim'].split('_')]
        self.block_dim = cfg['block_dim']

        mlp_dim_list = [self.pe_t.embed_length] + stem_dim_list + [self.block_dim]
        self.stem_t = NeRV_MLP(dim_list=mlp_dim_list, act=cfg['act'])

        # xy mapping
        xy_coord = torch.stack( 
            torch.meshgrid(
                torch.arange(self.fc_h) / self.fc_h, torch.arange(self.fc_w) / self.fc_w
            ), dim=0
        ).flatten(1, 2)  # [2, h*w]
        self.xy_coord = nn.Parameter(xy_coord, requires_grad=False)
        self.pe_xy = PositionalEncoding(
            pe_embed_b=cfg['xypos_b'], pe_embed_l=cfg['xypos_l']
        )
        
        self.stem_xy = NeRV_MLP(dim_list=[2 * self.pe_xy.embed_length, self.block_dim], act=cfg['act'])
        self.trans1 = TransformerBlock(
            dim=self.block_dim, heads=1, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        self.trans2 = TransformerBlock(
            dim=self.block_dim, heads=8, dim_head=64, mlp_dim=cfg['mlp_dim'], dropout=0., prenorm=False
        )
        if self.block_dim == self.fc_dim:
            self.toconv = nn.Identity()
        else:
            self.toconv = NeRV_MLP(dim_list=[self.block_dim, self.fc_dim], act=cfg['act'])
        
        # BUILD CONV LAYERS
        self.layers, self.head_layers, self.t_layers, self.norm_layers = [nn.ModuleList() for _ in range(4)]
        ngf = self.fc_dim
        for i, stride in enumerate(cfg['stride_list']):
            if i == 0:
                # expand channel width at first stage
                new_ngf = int(ngf * cfg['expansion'])
            else:
                # change the channel width for each stage
                new_ngf = max(ngf // (1 if stride == 1 else cfg['reduction']), cfg['lower_width'])
            
            self.t_layers.append(NeRV_MLP(dim_list=[128, 2*ngf], act=cfg['act']))
            self.norm_layers.append(nn.InstanceNorm2d(ngf, affine=False))
            
            if i == 0:
                self.layers.append(Conv_Up_Block(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            else:
                self.layers.append(NeRVBlock(ngf=ngf, new_ngf=new_ngf, stride=stride, bias=cfg['bias'], norm=cfg['norm'], act=cfg['act'], conv_type=cfg['conv_type']))
            ngf = new_ngf

            # build head classifier, upscale feature layer, upscale img layer 
            head_layer = [None]
            if cfg['sin_res']:
                if i == len(cfg['stride_list']) - 1:
                    head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
                else:
                    head_layer = None
            else:
                head_layer = nn.Conv2d(ngf, 3, 1, 1, bias=cfg['bias'])
            self.head_layers.append(head_layer)
        self.sigmoid = cfg['sigmoid']

        self.T_num = 20
        self.pe_t_manipulate = PositionalEncoding(pe_embed_b=cfg['pos_b_tm'], pe_embed_l=cfg['pos_l_tm'])
        self.t_branch = NeRV_MLP(dim_list=[self.pe_t_manipulate.embed_length, 128, 128], act=cfg['act'])

        self.loss = cfg['additional_loss'] if cfg.__contains__('additional_loss') else None
        self.loss_w = cfg['additional_loss_weight'] if cfg.__contains__('additional_loss_weight') else 1.0
        self.mse = nn.MSELoss()
    
    def fuse_t(self, x, t):
        # x: [B, C, H, W], normalized among C
        # t: [B, 2* C]
        f_dim = t.shape[-1] // 2
        gamma = t[:, :f_dim]
        beta = t[:, f_dim:]

        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        out = x * gamma + beta
        return out

    def forward_impl(self, input_id):
        t = input_id

        t_emb = self.stem_t(self.pe_t(t)) # [B, L]
        t_manipulate = self.t_branch(self.pe_t_manipulate(t))

        xy_coord = self.xy_coord
        x_coord = self.pe_xy(xy_coord[0])    # [h*w, C]
        y_coord = self.pe_xy(xy_coord[1])    # [h*w, C]
        xy_emb = torch.cat([x_coord, y_coord], dim=1)
        xy_emb = self.stem_xy(xy_emb).unsqueeze(0).expand(t_emb.shape[0], -1, -1)  # [B, h*w, L]

        xy_emb = self.trans1(xy_emb)
        # fuse t into xy map
        t_emb_list = [t_emb for i in range(xy_emb.shape[1])]
        t_emb_map = torch.stack(t_emb_list, dim=1)  # [B, h*w, L]
        emb = xy_emb * t_emb_map
        emb = self.toconv(self.trans2(emb))

        emb = emb.reshape(emb.shape[0], self.fc_h, self.fc_w, emb.shape[-1])
        emb = emb.permute(0, 3, 1, 2)
        output = emb

        out_list = []
        for layer, head_layer, t_layer, norm_layer in zip(self.layers, self.head_layers, self.t_layers, self.norm_layers):
            # t_manipulate
            output = norm_layer(output)
            t_feat = t_layer(t_manipulate)
            output = self.fuse_t(output, t_feat)
            # conv
            output = layer(output) 
            if head_layer is not None:
                img_out = head_layer(output)
                # normalize the final output iwth sigmoid or tanh function
                img_out = torch.sigmoid(img_out) if self.sigmoid else (torch.tanh(img_out) + 1) * 0.5
                out_list.append(img_out)

        return  out_list
    
    def forward(self, data):
        input_id = data['img_id']  # [B]
        batch_size = input_id.shape[0]

        output_list = self.forward_impl(input_id)  # a list containing [B or 2B, 3, H, W]

        if self.loss and self.training:
            b, c, h, w = output_list[-1].shape
            # NO USE
            grad_loss = 0.0
            return {
                "loss": grad_loss * self.loss_w,
                "output_list": output_list,
            }
        
        return output_list
