import copy
import math
import warnings
from datetime import time
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import resnet50
from collections import OrderedDict
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import MultiScaleDeformableAttention as MSDA


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class DeformableTransformer(nn.Module):

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=1024, dropout=0.3,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 3)

        self._reset_parameters()


    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    # def get_valid_ratio(self, mask):
    #     _,channel, H, W ,D= mask.shape
    #     valid_H = H
    #     valid_W = W
    #     valid_D = D
    #     valid_ratio_h = torch.tensor(valid_H/ H).repeat(_, 1)
    #     valid_ratio_w =torch.tensor( valid_W/ W).repeat(_, 1)
    #     valid_ratio_d =torch.tensor(valid_D/ D).repeat(_, 1)
    #     valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h,valid_ratio_d], 1)
    #     return valid_ratio
    def get_valid_ratio(self, mask):
        _, channel, H, W, D = mask.shape
        valid_H = H
        valid_W = W
        valid_D = D
        valid_ratio_h = torch.tensor(valid_H / H).repeat(_, 1)
        valid_ratio_w = torch.tensor(valid_W / W).repeat(_, 1)
        valid_ratio_d = torch.tensor(valid_D / D).repeat(_, 1)
        valid_ratio = torch.cat([valid_ratio_w, valid_ratio_h, valid_ratio_d], dim=1)  # 在第1维度上拼接
        return valid_ratio



    def forward(self, srcs,mask_prompt, pos_embeds, query_embed=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_prompt_in_transformer_pre= mask_prompt  #这里获取的是mask_prompt，pre表示没传入transformer，单纯为了登记，该forward函数是Transformer的前向传播函数
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs,  pos_embeds)):
            bs, c, h, w ,d= src.shape  #获取了src的尺寸大小
            spatial_shape = (h, w,d)   #记录一下特征图的大小
            spatial_shapes.append(spatial_shape)   #然后将记录好的特征图大小放在列表里面
            src = src.flatten(2).transpose(1,2)   #.flatten(2) 将 src 的形状从 (bs, c, h, w, d) 变为 (bs, c, h * w * d)   flatten(2)在第二个维度开始展平，然后transpose交换位置

            pos_embed = pos_embed.flatten(2).transpose(1,2)  #跟上面同理
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    #给位置编码加上层级编码
            lvl_pos_embed_flatten.append(lvl_pos_embed)  #把全部位置编码加层级编码后的结合在一起
            src_flatten.append(src)  #将全部的特征图结合在一起

        src_flatten = torch.concat(src_flatten, dim=1)  #将特征图全部展平后的结果，concat在一起
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  #同理


        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)  #记录concat后的每一层所在的第一个索引

        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        #valid_ratios = torch.stack([self.get_valid_ratio(m) for m in srcs], 1)
        #valid_ratios = torch.cat([self.get_valid_ratio(m).unsqueeze(1) for m in srcs], 1)
        ratios_list = [self.get_valid_ratio(m) for m in srcs]
        valid_ratios = torch.stack(ratios_list, dim=1).to(device)
        # encoderv
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten)    #【2，11236，256】

        # prepare input for decoder
        bs, _, c = memory.shape  #batch-size =2    _=11236    c=256




        if self.two_stage:#不进入这个判断语句
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, spatial_shapes)
            # hack implementation for two-sta ge Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)# 不走这里
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals# 不走这里
            topk = self.two_stage_num_proposals# 不走这里
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]# 不走这里
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))# 不走这里
            topk_coords_unact = topk_coords_unact.detach()# 不走这里
            reference_points = topk_coords_unact.sigmoid()# 不走这里
            init_reference_out = reference_points# 不走这里
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))# 不走这里
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)  # 不走这里



        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)  #本来是300，512  切开成300 256*2 一个给tgt 一个给object queries


            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)   #(2,300,256)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)#(2,300,256)


            query_embed=query_embed+mask_prompt_in_transformer_pre
            
            reference_points = self.reference_points(query_embed).sigmoid()  #(2,300,3)
            init_reference_out = reference_points  #留用

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        return hs, init_reference_out, inter_references_out, None, None


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.3, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    # def get_reference_points(spatial_shapes, valid_ratios, device):
    #     reference_points_list = []
    #     for lvl, (H_, W_,D_) in enumerate(spatial_shapes):
    #
    #         ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
    #                                       torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
    #         ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
    #         ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
    #         ref = torch.stack((ref_x, ref_y), -1)
    #         reference_points_list.append(ref)
    #     reference_points = torch.cat(reference_points_list, 1)
    #     reference_points = reference_points[:, :, None] * valid_ratios[:, None]
    #     return reference_points

    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_, D_) in enumerate(spatial_shapes):
            #匹配一下数据以适应meshgrid
            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            # print("ref_z.reshape(1,-1) 的形状是：", ref_z.reshape(1, -1).shape)
            # print("valid_ratios[:, None, lvl, 2] 的形状是：", valid_ratios[:, None, lvl, 2].shape)
            # 归一化
            ref_z = ref_z.reshape(1,-1) / (valid_ratios[:, None, lvl, 2] * D_)
            ref_y = ref_y.reshape(1,-1)/ (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(1,-1)/ (valid_ratios[:, None, lvl, 0] * W_)

            # 堆叠成 [1, H_*W_*D_, 3] 的张量
            ref = torch.stack((ref_x, ref_y, ref_z), -1)
            reference_points_list.append(ref)

        # 合并所有尺度的参考点
        reference_points = torch.cat(reference_points_list, 1)

        # 调整参考点的位置以适应实际输入图像尺寸，边框对其
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points



    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None):
       
        output = src
        valid_ratios = valid_ratios.to(src.device)  # 添加这行代码
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index)

        return output


class MSDeformAttnFunction(Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                im2col_step):
        ctx.im2col_step = im2col_step
        output = MSDA.ms_deform_attn_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
            ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = \
            MSDA.ms_deform_attn_backward(
                value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                grad_output, ctx.im2col_step)

        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None
    

class MSDeformAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 3)
        #self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

       #self._reset_parameters()

    # def _reset_parameters(self):
    #     constant_(self.sampling_offsets.weight.data, 0.)
    #     thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
    #     grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    #     grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
    #     for i in range(self.n_points):
    #         grid_init[:, :, i, :] *= i + 1
    #     with torch.no_grad():
    #         self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    #     constant_(self.attention_weights.weight.data, 0.)
    #     constant_(self.attention_weights.bias.data, 0.)
    #     xavier_uniform_(self.value_proj.weight.data)
    #     constant_(self.value_proj.bias.data, 0.)
    #     xavier_uniform_(self.output_proj.weight.data)
    #     constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):

        N, Len_q, _ = query.shape  #N:Batch-siez  len_q: queries的长度  _:256
        N, Len_in, _ = input_flatten.shape   #len_q: queries的长度
        # assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:,1]).sum() == Len_in  #检查

        assert (input_spatial_shapes.prod(dim=1)).sum() == Len_in

        value = self.value_proj(input_flatten)  #输入256 输出也是256
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)  # 例子中是：2，11236，8个头，每个头负责32个维度  【2，11236，8，32】
        #sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)  # 【2，11236,8,4,4,2】
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 3)#要适配3d  【2，11236,8,4,4,3】  [2,11236,384]
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points) #【2，11236，128】  ---》  【2，11236，8个头，16==特征图4个，key为4个】
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)  #2，11236，8个头，4个features，4个key
        # N, Len_q, n_heads, n_levels, n_points, 2

        if reference_points.shape[-1] == 3:
            # offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            # sampling_locations = reference_points[:, :, None, :, None, :] \
            #                      + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 2], input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :]  + sampling_offsets / offset_normalizer[None, None, None, :, None, :]




        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2]  + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))


       

        #output = PyTorchDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step) #pytorch版本

        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)  #cuda版本

        output = self.output_proj(output)
        return output

class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.3, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):  # query_pos就是 query_embed
        # self attention

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)  #[0] 是为了取出attn_output，不需要[1]attn_weights   得到的是【300，2，256】  然后再transpose一下位置
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),  reference_points,src, src_spatial_shapes, level_start_index, src_padding_mask)
                                # Q是tgt,+query_pos。
                                # k是query_pos(初始话的query_object+mask) 
                        
                                #value 是memory
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class DeformableTransformerDecoder(nn.Module):
    def  __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None):

                #tgt  内容信息，一开始这里是query embed中生成的  【2，300，256】
                #reference_points:查询位置坐标的偏移量   【2，300，3】
                #src memory，用于生成v的部分 【2，11236，256】
                #src_spatial_shapes:用于记录4个特征图的尺寸
                #src_level_start_index:4个特征图第一个位置的索引
                #src_valid_ratios:每个特征图的缩放比例
        output = tgt

        intermediate = []     #存放每个decoder  layer的输出
        intermediate_reference_points = []  # 存放每个一个decoder使用的reference points
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 3
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index)

            # hack implementation for iterative bounding box refinement
            # if self.bbox_embed is not None:  #这个不执行
            #     tmp = self.bbox_embed[lid](output)#这个不执行
            #     if reference_points.shape[-1] == 4:#这个不执行
            #         new_reference_points = tmp + inverse_sigmoid(reference_points)#这个不执行
            #         new_reference_points = new_reference_points.sigmoid()#这个不执行
            #     else:#这个不执行
            #         assert reference_points.shape[-1] == 2#这个不执行
            #         new_reference_points = tmp#这个不执行
            #         new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)#这个不执行
            #         new_reference_points = new_reference_points.sigmoid()#这个不执行
            #     reference_points = new_reference_points.detach()#这个不执行

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)   #stack后的尺寸是[6,2,300,2]

        return output, reference_points

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
class IntermediateLayerGetter(nn.Module):
    """
    Module wrapper that returns intermediate layers from a model.
    It uses an OrderedDict to maintain the order of layers.
    """
    def __init__(self, model, return_layers):
        super().__init__()
        assert set(return_layers).issubset([name for name, _ in model.named_children()]), \
            "return_layers are not present in model"
        self.return_layers = return_layers
        self.model = model

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.model.named_children():
            x = module(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
              #  print(f"Output of {name}: {x.shape}")  # Print the shape of each output
            if len(out) == len(self.return_layers):
                break
        return out

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list):
        # Sequential类中的self[0]通常代表第一个被添加的子模块
        xs = self[0](tensor_list)
        out: List[torch.Tensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.dtype))

        # 所以最后backbone的输出就是 所需要的不同尺度特征图以及位置编码
        return out, pos
class CustomResNet3D(nn.Module):
    def __init__(self, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        backbone = resnet50(spatial_dims=3, n_input_channels=1, num_classes=2)

        # 设置是否训练 backbone 的特定层
        for name, parameter in backbone.named_parameters():
            if not train_backbone or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name):
                parameter.requires_grad_(False)

        # 根据是否需要返回中间层的特征图来设置 return_layers
        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]  # 这些步长值可能需要根据具体的3D ResNet50架构进行调整
            self.num_channels = [512, 1024, 2048]  # 根据 ResNet50 的实际通道数进行调整
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]

        # 使用 NetAdapter 来选择性地获取中间层输出
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, x):
        xs = self.body(x)
        out = {}
        for name, x in xs.items():
            # 此处可以根据需要对输出特征图做进一步操作
            out[name] = x
        #print(type(out))

        return out

class PositionEmbeddingSine3D(nn.Module):
    """
    This class extends the position embedding to 3D and adjusts num_pos_feats to 85.
    """
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list

        # Create position indices for 3D
        z_embed = torch.arange(x.shape[2], device=x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        y_embed = torch.arange(x.shape[3], device=x.device).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        x_embed = torch.arange(x.shape[4], device=x.device).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1, :, :][:, None, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1, :][:, :, None, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1][:, :, :, None] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t //2) / self.num_pos_feats)

        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t

        pos_z_sin = pos_z[:, :, :, :, 0::2].sin()# 增加一个新的维度来准备连接
        pos_z_cos = pos_z[:, :, :, :, 1::2].cos()
        pos_z = torch.cat((pos_z_sin, pos_z_cos), dim=4).flatten(4)

        # 对于 pos_y，同样的方法
        pos_y_sin = pos_y[:, :, :, :, 0::2].sin()
        pos_y_cos = pos_y[:, :, :, :, 1::2].cos()
        pos_y = torch.cat((pos_y_sin, pos_y_cos), dim=4).flatten(4)

        # 对于 pos_x
        pos_x_sin = pos_x[:, :, :, :, 0::2].sin()
        pos_x_cos = pos_x[:, :, :, :, 1::2].cos()
        pos_x = torch.cat((pos_x_sin, pos_x_cos), dim=4).flatten(4)


        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)

        # Expand to match the required channel dimension
        if pos.shape[1] < 256:
            torch.manual_seed(0)  # Set random seed
            random_pad = torch.randn((pos.shape[0], 256 - pos.shape[1], pos.shape[2], pos.shape[3], pos.shape[4]), device=pos.device)
            pos = torch.cat([pos, random_pad], dim=1)

        return pos

class MASKPositionEmbeddingSine3D(nn.Module):
    """
    This class extends the position embedding to 3D and adjusts num_pos_feats to 85.
    """
   
    def __init__(self, num_pos_feats, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list

        # Create position indices for 3D
        z_embed = torch.arange(x.shape[2], device=x.device).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        y_embed = torch.arange(x.shape[3], device=x.device).unsqueeze(0).unsqueeze(1).unsqueeze(3).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        x_embed = torch.arange(x.shape[4], device=x.device).unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(x.shape[0], x.shape[2], x.shape[3], x.shape[4])

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1, :, :][:, None, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1, :][:, :, None, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1][:, :, :, None] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t //2) / self.num_pos_feats)

        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t

        pos_z_sin = pos_z[:, :, :, :, 0::2].sin()# 增加一个新的维度来准备连接
        pos_z_cos = pos_z[:, :, :, :, 1::2].cos()
        pos_z = torch.cat((pos_z_sin, pos_z_cos), dim=4).flatten(4)

        # 对于 pos_y，同样的方法
        pos_y_sin = pos_y[:, :, :, :, 0::2].sin()
        pos_y_cos = pos_y[:, :, :, :, 1::2].cos()
        pos_y = torch.cat((pos_y_sin, pos_y_cos), dim=4).flatten(4)

        # 对于 pos_x
        pos_x_sin = pos_x[:, :, :, :, 0::2].sin()
        pos_x_cos = pos_x[:, :, :, :, 1::2].cos()
        pos_x = torch.cat((pos_x_sin, pos_x_cos), dim=4).flatten(4)


        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)

        # Expand to match the required channel dimension
        # if pos.shape[1] < 256:
        #     torch.manual_seed(0)  # Set random seed
        #     random_pad = torch.randn((pos.shape[0], 256 - pos.shape[1], pos.shape[2], pos.shape[3], pos.shape[4]), device=pos.device)
        #     pos = torch.cat([pos, random_pad], dim=1)

        return pos

# class MASKPositionEmbeddingSine3D(nn.Module):
#     """
#     This class extends the position embedding to 3D and matches the input shape.
#     The output will have the same shape as the input: (batch_size, 1, depth, height, width).
#     """
#     def __init__(self, temperature=10000, normalize=False, scale=None):
#         super().__init__()
#         self.temperature = temperature
#         self.normalize = normalize
#         if scale is not None and not normalize:
#             raise ValueError("normalize should be True if scale is passed")
#         self.scale = 2 * math.pi if scale is None else scale

#     def forward(self, tensor_list):
#         x = tensor_list
#         batch_size, _, depth, height, width = x.shape
        
#         # Create position indices for 3D
#         z_embed = torch.arange(depth, device=x.device).float()
#         y_embed = torch.arange(height, device=x.device).float()
#         x_embed = torch.arange(width, device=x.device).float()

#         if self.normalize:
#             eps = 1e-6
#             z_embed = (z_embed / (depth + eps)) * self.scale
#             y_embed = (y_embed / (height + eps)) * self.scale
#             x_embed = (x_embed / (width + eps)) * self.scale

#         # Generate position embeddings with a single channel
#         pos_z = z_embed.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(batch_size, 1, depth, height, width)
#         pos_y = y_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, 1, depth, height, width)
#         pos_x = x_embed.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, depth, height, width)

#         # Combine the position embeddings into one tensor
#         pos = (pos_z + pos_y + pos_x) / 3  # Average the embeddings

#         return pos



class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, transformer,position_mask_embedding,num_classes=2, num_queries=512, num_feature_levels=4,
                 aux_loss=True, with_box_refine=False, two_stage=False):

        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim*2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)   # self.strides = [8, 16, 32]
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv3d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv3d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList( [
                nn.Sequential(
                    nn.Conv3d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        self.position_mask_embedding = position_mask_embedding

        self.mask_downscaling = nn.Sequential(
            nn.Conv3d(96, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 4, 64, 64, 64]
            nn.LayerNorm([4, 64, 64, 64]),
            nn.GELU(),

            nn.Conv3d(4, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 16, 32, 32, 32]
            nn.LayerNorm([16, 32, 32, 32]),
            nn.GELU(),

            nn.Conv3d(16, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 64, 16, 16, 16]
            nn.LayerNorm([64,16,16,16]),
            nn.GELU(),

            nn.Conv3d(64, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 256, 8, 8, 8]
            nn.LayerNorm([256, 8, 8, 8]),


            nn.Flatten(start_dim=2),  # 将[2, 256, 8*8*8]展平为[2, 256, 512]
        )
        self.mask_downscaling_full_zero = nn.Sequential(
            nn.Conv3d(1, 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 4, 64, 64, 64]
            nn.LayerNorm([4, 64, 64, 64]),
            nn.GELU(),

            nn.Conv3d(4, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 16, 32, 32, 32]
            nn.LayerNorm([16, 32, 32, 32]),
            nn.GELU(),

            nn.Conv3d(16, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 64, 16, 16, 16]
            nn.LayerNorm([64,16,16,16]),
            nn.GELU(),

            nn.Conv3d(64, 256, kernel_size=(2, 2, 2), stride=(2, 2, 2)),  # [2, 256, 8, 8, 8]
            nn.LayerNorm([256, 8, 8, 8]),


            nn.Flatten(start_dim=2),  # 将[2, 256, 8*8*8]展平为[2, 256, 512]
        )
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])

            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        self.final_fc = nn.Linear(num_queries * hidden_dim, num_classes)

    def extract_top_k_predictions(self,outputs_classes, k=10):
        num_layers = len(outputs_classes)
        batch_size = outputs_classes[0].shape[0]
        num_queries = outputs_classes[0].shape[1]
        num_classes = outputs_classes[0].shape[2]

        # 初始化存储最大前 k 个预测值的列表
        top_k_values_class_0 = [[] for _ in range(batch_size)]
        top_k_values_class_1 = [[] for _ in range(batch_size)]

        for batch_idx in range(batch_size):
            for layer_idx in range(num_layers):
                # 提取当前 batch 和当前层的预测值
                predictions = outputs_classes[layer_idx][batch_idx]

                # 提取类别 0 和类别 1 的预测值
                class_0_scores = predictions[:, 0]
                class_1_scores = predictions[:, 1]

                # 获取最大前 k 个预测值的索引
                top_k_indices_class_0 = torch.topk(class_0_scores, k=k).indices
                top_k_indices_class_1 = torch.topk(class_1_scores, k=k).indices

                # 提取最大前 k 个预测值
                top_k_values_class_0[batch_idx].extend(class_0_scores[top_k_indices_class_0])
                top_k_values_class_1[batch_idx].extend(class_1_scores[top_k_indices_class_1])

        # 计算每个 batch 和每个类别的平均值
        avg_values = torch.zeros((batch_size, num_classes))
        for batch_idx in range(batch_size):
            avg_values[batch_idx, 0] = torch.mean(torch.stack(top_k_values_class_0[batch_idx]))
            avg_values[batch_idx, 1] = torch.mean(torch.stack(top_k_values_class_1[batch_idx]))

        return avg_values



    
    def forward(self, samples,mask_prompt):


#         def forward(self, samples,mask_prompt):




#         # if not isinstance(samples, NestedTensor):
#         #     samples = nested_tensor_from_tensor_list(samples)
#         features, pos = self.backbone(samples)
#         mask_pos = self.position_mask_embedding(mask_prompt)
#         mask_prompt = mask_prompt + mask_pos
#         mask_prompt=self.mask_downscaling(mask_prompt)
#         mask_prompt=mask_prompt.permute(0,2,1)

        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        if mask_prompt.sum() != 0:
            mask_pos = self.position_mask_embedding(mask_prompt)
            mask_prompt = mask_prompt + mask_pos #位置编码跟mask相结合
            mask_prompt = self.mask_downscaling(mask_prompt) #降采样匹配维度
        else:
        # 如果mask_prompt全为0，跳过
            mask_prompt = mask_prompt  
            mask_prompt = self.mask_downscaling_full_zero(mask_prompt)#降采样匹配维度
       
        mask_prompt = mask_prompt.permute(0, 2, 1)
       
        srcs = []

        for l, feat in enumerate(features):
            src = feat  #当前的特征图
            srcs.append(self.input_proj[l](src))
            # masks.append(mask)
            # assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1])
                else:
                    src = self.input_proj[l](srcs[-1])

                pos_l = self.backbone[1](src).to(src.dtype)#生成位置编码
                srcs.append(src)

                pos.append(pos_l)

        query_embeds = None


        if not self.two_stage:
            query_embeds = self.query_embed.weight


        hs , init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact= self.transformer(srcs,mask_prompt, pos, query_embeds)

        #hs的shape是[6,2,300,256]  6是6个decoder layer  2是bs   300个queries 256是dimension
        #init_reference的shape是[2,300,2]  2是bs  300个queries 自动生成的，2是xy
        #init_references的shape是[6,2,300,2]   6个decoder里面的reference

        layer_outputs = []

        for lvl in range(hs.shape[0]):  #取出6个layer
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            #reference = inverse_sigmoid(reference)  #还原回未归一化后的左边，应该是用来做画框操作的

            hs_lvl = hs[lvl]

            hs_lvl_flat = hs_lvl.view(hs_lvl.size(0), -1)  # 形状为 [batch_size, num_queries * hidden_dim]
            layer_output = self.final_fc(hs_lvl_flat)  # 形状为 [batch_size, num_classes]
            layer_outputs.append(layer_output)

        all_layer_outputs = torch.stack(layer_outputs)
        final_output = torch.mean(all_layer_outputs, dim=0)

        return final_output







    @torch.jit.unused
    def _set_aux_loss(self, outputs_class):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a}
                for a in zip(outputs_class[:-1])]

class PyTorchDeformAttnFunction(Function):
    # pytorch实现
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):

        bch, Len_in, n_heads, D_ = value.shape   # bch=batch_sieze :2    len_in =queries=11236 n_heads=8 ,d=32
        _, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape  # batch_size ,queries,heads,feature_map,key_points=[2,11236,8,4,4,3]
        value_list = value.split([H_ * W_ * Depth_ for H_, W_,Depth_ in value_spatial_shapes], dim=1)  #拆分获得每个feature_map的大小的列表
        sampling_grids = 2 * sampling_locations - 1   #坐标做适配
        sampling_value_list = []  #存放是K的位置所对应的value的值域

        for lid_, (H_, W_,Depth_) in enumerate(value_spatial_shapes):
            #batch-size,H*W*Depth,heads,dimension   ----->   batch-size,H*W*Depth,heads*dimension   ----->  batch-size,heads*dimension,H*W*Depth   -----> batch-size*heads,dimension,H,W,Depth
            value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(bch*n_heads, D_, H_, W_,Depth_)

            #batch-size,queries_len,heads,key_points,axis=2   --->  batch-size,heads,queries_len,key_points,axis=2  --->  batch-size*heads,queries_len,key_points,axis=2
            sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1 , 2).flatten(0, 1)
            sampling_grid_l_ = sampling_grid_l_.unsqueeze(3)
            #batch-size*heads,dimension,queries_len,key_points
            sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_, mode='bilinear', padding_mode='zeros', align_corners=False)

            sampling_value_list.append(sampling_value_l_)  #每一个都是【16，32，299520，4，1】


        #(batch-size,queries,heads,features_layers_number,key-points)    --->   (batch-size,heads,queries,features_layers_number,key-points)   --->   (batch-size*heads , 1 ,queries, features_layers_number * key-points)
        attention_weights = attention_weights.transpose(1, 2).reshape(bch*n_heads, 1, Len_q, n_levels*n_points)  # 源代码里面是[16，1，11236，16]  我这里是【16，1，299520，16】

        output = torch.stack(sampling_value_list, dim=-2)  #
        output=output.flatten(-3)  #我这里的shape是[16，32，299520，4，4]

        output = output * attention_weights  #output的shape是[16，32，299520，4，4]，attention_weights的shape是源代码里面是[16，1，11236，16]
        output = output.sum(-1).view(bch, n_heads*D_, Len_q)

        # output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(bch, n_heads * D_,Len_q)
        return output.transpose(1, 2).contiguous()

# if __name__ == '__main__':
#     # 使用模型示例
#     position_embedding = PositionEmbeddingSine3D(num_pos_feats=85,normalize=True)
#     backbone = CustomResNet3D(train_backbone=True, return_interm_layers=True)
#     model = Joiner(backbone, position_embedding)  #它就是backbone
#     input_tensor = torch.rand(2, 1, 256, 256, 256)  # 假设输入是一个1通道的3D体积

#     output_features = model(input_tensor)
if __name__ == '__main__':
    backbone = CustomResNet3D(train_backbone=True, return_interm_layers=True).to(device)
    position_embedding = PositionEmbeddingSine3D(num_pos_feats=85, normalize=True).to(device)
    mask_position_embedding=MASKPositionEmbeddingSine3D(num_pos_feats=32, normalize=True).to(device)
    
    input_tensor = torch.rand(2, 1, 128, 128, 128).to(device)
    mask_tensor = torch.rand(2, 1, 128, 128, 128).to(device)
    combine_b_p = Joiner(backbone, position_embedding).to(device)
    # 初始化 Deformable DETR 模型
    transformer = DeformableTransformer(d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                        dim_feedforward=1024, dropout=0.3, activation="relu",
                                        return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                                        enc_n_points=4, two_stage=False, two_stage_num_proposals=300).to(device)
    detr_model = DeformableDETR(backbone=combine_b_p, transformer=transformer, position_mask_embedding=  mask_position_embedding, num_classes=2, num_queries=4096,
                                num_feature_levels=4, aux_loss=True, with_box_refine=False, two_stage=False).to(device)
    

    # 创建一个随机输入张量
   
    # 将输入张量传递给模型
    outputs = detr_model(input_tensor,mask_tensor)

    # 打印输出结果
    print(outputs)