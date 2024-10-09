import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.datasets.TITAN import ACT_SET_TO_N_CLS
from .backbones import create_backbone




def gather_nd(params, indices):
    """ The same as tf.gather_nd but batched gather is not supported yet.
    indices is an k-dimensional integer tensor, best thought of as a (k-1)-dimensional tensor of indices into params, where each element defines a slice of params:

    output[\\(i_0, ..., i_{k-2}\\)] = params[indices[\\(i_0, ..., i_{k-2}\\)]]

    Args:
        params (Tensor): "n" dimensions. shape: [x_0, x_1, x_2, ..., x_{n-1}]
        indices (Tensor): "k" dimensions. shape: [y_0,y_2,...,y_{k-2}, m]. m <= n.

    Returns: gathered Tensor.
        shape [y_0,y_2,...y_{k-2}] + params.shape[m:]

    """
    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]    # (num_samples, ...)
    return output.reshape(out_shape).contiguous()

def cond(condition, fn1, fn2):
    if condition:
        return fn1()
    else:
        return fn2()

def softmax(logits, scope=None):
    """a flatten and reconstruct version of softmax."""
    out = torch.softmax(logits, dim=-1)
    return out

def softsel(target, logits, use_sigmoid=False, scope=None):
    """Apply attention weights."""
    if use_sigmoid:
        a = torch.sigmoid(logits)
    else:
        a = softmax(logits)  # shape is the same
    target_rank = len(target.shape)
    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
    # second last dim
    return (torch.unsqueeze(a, -1) * target).sum(target_rank - 2)

def focal_attention(query, context, use_sigmoid=False):
    """Focal attention layer.

  Args:
    query : [N, dim1]
    context: [N, num_channel, T, dim2]
    use_sigmoid: use sigmoid instead of softmax
    scope: variable scope

  Returns:
    Tensor
  """

    # Tensor dimensions, so pylint: disable=g-bad-name
    _, d = query.shape
    _, K, _, d2 = context.shape
    assert d == d2

    T = context.shape[2]

    # [N,d] -> [N,K,T,d]
    query_aug = torch.unsqueeze(
        torch.unsqueeze(query, 1), 1).repeat(1, K, T, 1)

    # cosine simi
    query_aug_norm = F.normalize(query_aug, p=2, dim=-1)
    context_norm = F.normalize(context,p=2, dim=-1)
    # [N, K, T]
    a_logits = torch.multiply(query_aug_norm, context_norm).sum(3)

    a_logits_maxed = a_logits.amax(2)  # [N,K]

    attended_context = softsel(softsel(context, a_logits,
                                       use_sigmoid=use_sigmoid), a_logits_maxed,
                               use_sigmoid=use_sigmoid)
    # print(query.mean(),context.mean(), a_logits.mean(), a_logits_maxed.mean(),attended_context.mean())
    return attended_context

def concat_states(state_tuples, dim):
    """Concat LSTM states."""
    return LSTMStateTuple(c=torch.cat([s.c for s in state_tuples],
                                      dim=dim),
                          h=torch.cat([s.h for s in state_tuples],
                                      dim=dim))

class Conv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel, padding='SAME', stride=1,
                 activation=torch.nn.Identity, add_bias=True, data_format='NHWC',
                 w_init=None, scope='conv'):
        super(Conv2d, self).__init__(
            in_channel, out_channel, kernel, stride=stride, bias=add_bias
        )
        self.scope = scope
        self.activation = activation()
        self.data_format = data_format
        if w_init is None:
            nn.init.kaiming_normal_(self.weight)
            # nn.init.constant_(self.weight, 1)
        else:
            w_init(self.weight)
        if add_bias:
            nn.init.constant_(self.bias,0)

    def calc_same_pad(self, h, w,  k, s):
        out_height = np.ceil(float(h) / float(s[0]))
        out_width = np.ceil(float(w) / float(s[1]))

        if (h % s[0] == 0):
            pad_along_height = max(k[0] - s[0], 0)
        else:
            pad_along_height = max(k[0] - s[0]- (h % s[0]), 0)
        if (w % s[1] == 0):
            pad_along_width = max(k[1] - s[1], 0)
        else:
            pad_along_width = max(k[1]  - (w % s[1]), 0)
        pad_bottom = pad_along_height // 2
        pad_top = pad_along_height - pad_bottom
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return pad_top, pad_bottom, pad_left, pad_right

    def forward(self, x):

        if self.data_format == "NHWC":
            x = x.permute(0, 3, 1, 2)
        N, C, H, W = x.shape
        ih, iw = x.size()[-2:]
        t,b,l, r = self.calc_same_pad(H, W, self.kernel_size, self.stride)

        x = F.pad(
            x, [l,r, t,b],
        )
        x = super().forward(
            x,
        )
        x = self.activation(x)
        if self.data_format == "NHWC":
            x = x.permute(0, 2, 3, 1)
        return x

class Linear(nn.Module):
    def __init__(self, input_size, output_size, scope="", add_bias=True, 
                 activation=nn.Identity):
        super(Linear, self).__init__()
        self.linear = nn.Linear(
            input_size,
            output_size,
            bias=add_bias
        )
        self.scope = scope
        self.activation = activation()

        nn.init.trunc_normal_(self.linear.weight, std=0.1)
        # nn.init.constant_(self.linear.weight, 1.0)
        if add_bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x


class LSTMStateTuple:
    def __init__(self, h, c):
        self.h = h
        self.c = c

class LSTMCell(nn.Module):
    def __init__(self,input_keep_prob=1, *args, **kwargs):
        super(LSTMCell, self).__init__()
        self.lstm = nn.LSTMCell(*args, **kwargs)
        self.input_keep_prob = nn.Dropout(1-input_keep_prob)

    def forward(self, x, state=None):
        x = self.input_keep_prob(x)
        if state is not None:
            h, c = state.h, state.c
            (h, c) = self.lstm(x, (h, c))
            return LSTMStateTuple(h, c)
        else:
            (h, c) = self.lstm(x)
            return LSTMStateTuple(h, c)

class LSTM(nn.Module):
    def __init__(self, scope="", input_keep_prob=1, *args, **kwargs):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(*args, **kwargs, batch_first=True)
        # for name, param in self.lstm.named_parameters():
        #     if "weight" in name:
        #         nn.init.xavier_normal_(param)
        #         nn.init.constant_(param, 1.0)
        #     elif "bias" in name:
        #         nn.init.constant_(param, 0.0)
        self.scope = scope
        # consistent with dropout wrapper
        self.input_keep_prob = nn.Dropout(1 - input_keep_prob)

    def forward(self, x, state=None):
        x = self.input_keep_prob(x)
        if state is not None:
            h, c = state
            output, (h, c) = self.lstm(x, (h, c))
        else:
            output, (h, c) = self.lstm(x)
        return output, LSTMStateTuple(h[0], c[0])
    

class Next(nn.Module):
    def __init__(self,
                 obs_len=4,
                 pred_len=4,
                 action_sets=['cross'],
                 img_backbone_name='vgg16',
                 n_seg=4,
                 n_neighbor_cls=1
                 ) -> None:
        super().__init__()
        self.action_sets = action_sets
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.img_backbone_name = img_backbone_name
        self.kp_size = 17
        self.emb_size = 128
        self.scene_conv_dim = 64
        self.enc_hidden_size = 256
        self.dec_hidden_size = 256
        self.person_feat_dim = 512
        self.box_emb_size = 64
        self.keep_prob = 0.7
        self.activation_func = nn.Tanh
        self.n_seg = n_seg
        self.n_neighbor_cls = n_neighbor_cls
        feature_size = self.enc_hidden_size * 5

        # traj
        self.enc_xy_emb = Linear(4, output_size=self.emb_size,
                                 activation=self.activation_func,
                                 add_bias=True, scope="enc_xy_emb")
        self.enc_traj = LSTM(
            input_size=self.emb_size,  # 128
            hidden_size=self.enc_hidden_size,  # 256
            input_keep_prob=self.keep_prob  # 0.7
        )
        # appearance
        self.img_backbone = create_backbone(self.img_backbone_name)
        self.enc_person = LSTM(
            input_size=self.person_feat_dim,  # 256
            hidden_size=self.enc_hidden_size,
            input_keep_prob=self.keep_prob
        )
        # person scene
        self.enc_personscene = LSTM(
            input_size=self.scene_conv_dim,  # 64
            hidden_size=self.enc_hidden_size,
            input_keep_prob=self.keep_prob
        )
        # pose
        self.enc_kp = LSTM(
                input_size=self.emb_size,
                hidden_size=self.enc_hidden_size,
                input_keep_prob=self.keep_prob
            )
        
        self.enc_other = LSTM(
            input_size=self.box_emb_size * 2,
            hidden_size=self.enc_hidden_size,
            input_keep_prob=self.keep_prob
        )
        self.dec_cell_traj = LSTMCell(
                input_size=self.emb_size + self.enc_hidden_size,
                hidden_size=self.dec_hidden_size,
                input_keep_prob=self.keep_prob
            )
        
        conv_dim = self.scene_conv_dim
        self.conv2 = Conv2d(
            in_channel=4,
            out_channel=conv_dim,
            kernel=3,
            stride=2, activation=self.activation_func,
            add_bias=True, scope='conv2'
        )

        self.conv3 = Conv2d(
            in_channel=conv_dim,
            out_channel=conv_dim,
            kernel=3,
            stride=2, activation=self.activation_func,
            add_bias=True, scope='conv3')

        self.kp_emb = Linear(input_size=self.kp_size * 2, output_size=self.emb_size, add_bias=True,
                             activation=self.activation_func)

        self.other_box_geo_emb = Linear(
            input_size=4,
            add_bias=True,
            activation=self.activation_func, output_size=self.box_emb_size,
            scope='other_box_geo_emb')

        self.other_box_class_emb = Linear(
            input_size=self.n_neighbor_cls,
            add_bias=True,
            activation=self.activation_func, output_size=self.box_emb_size,
            scope='other_box_class_emb')
        
        

        self.out_xy_mlp2 = Linear(
            input_size=self.dec_hidden_size,
            output_size=4,
            add_bias=False, scope='out_xy_mlp2')

        self.xy_emb_dec = Linear(
            input_size=4,
            output_size=self.emb_size,
            activation=self.activation_func, add_bias=True,
            scope='xy_emb_dec'
        )

        self.future_act = {}
        for act_set in self.action_sets:
            self.future_act[act_set] = Linear(
                input_size=feature_size,
                output_size=ACT_SET_TO_N_CLS[act_set], 
                add_bias=False)
        self.future_act = nn.ModuleDict(self.future_act)
        
    def decode(self,
               first_input, 
               enc_last_state, 
               enc_h, 
               pred_length,
               rnn_cell):

        curr_cell_state = enc_last_state
        decoder_out_ta = [first_input]

        for i in range(pred_length):
            curr_input_xy = decoder_out_ta[-1]

            xy_emb = self.xy_emb_dec(curr_input_xy)
            # print(i, torch.mean(curr_cell_state.h))
            attended_encode_states = focal_attention(
                curr_cell_state.h, enc_h, use_sigmoid=False)
            # print(i, torch.mean(curr_cell_state.h), torch.mean(enc_h), torch.mean(attended_encode_states))
            rnn_input = torch.cat(
                [xy_emb, attended_encode_states], dim=1)

            next_cell_state = rnn_cell(rnn_input, curr_cell_state)

            decoder_out_ta.append(self.hidden2xy(next_cell_state.h))
            curr_cell_state = next_cell_state

        decoder_out = torch.stack(decoder_out_ta[1:],dim=0)  # [T2,N,h_dim]
        # [N,T2,h_dim]
        decoder_out = decoder_out.permute(1, 0, 2)
        # decoder_out = self.hidden2xy(
        #     decoder_out_h)
        # print("decoder", decoder_out[0:2], decoder_out.shape, decoder_out.mean())
        return decoder_out
        
    
    def hidden2xy(self, lstm_h):
        """Hiddent states to xy coordinates."""
        # Tensor dimensions, so pylint: disable=g-bad-name
        out_xy = self.out_xy_mlp2(lstm_h)
        return out_xy
    
    def forward(self,
                batch):
        
        KP = 17
        N = batch[list(batch.keys())[0]].size(0)

        # encode traj
        traj_xy_emb_enc = self.enc_xy_emb(batch['traj'])  # B T 4 --> B T d
        traj_obs_enc_h, traj_obs_enc_last_state = self.enc_traj(
            traj_xy_emb_enc)
        enc_h_list = [traj_obs_enc_h]
        enc_last_state_list = [traj_obs_enc_last_state]

        # encode appearance
        appe = batch['img']
        B,C,T,H,W = appe.size()
        appe_emb = self.img_backbone(appe.permute(0,2,1,3,4).reshape(B*T,C,H,W))
        _, C1,H1,W1 = appe_emb.size()
        appe_emb = appe_emb.reshape(B,T,C1,H1,W1).mean([3,4])  # B T C1
        appe_h, appe_last = self.enc_person(appe_emb)
        enc_h_list.append(appe_h)
        enc_last_state_list.append(appe_last)

        # encode pose
        # B 2 T nj --> B T 2 nj
        obs_kp = batch['sklt'].permute(0,2,3,1).reshape(B, T, self.kp_size * 2)
        obs_kp = self.kp_emb(obs_kp)
        kp_obs_enc_h, kp_obs_enc_last_state = self.enc_kp(obs_kp)
        enc_h_list.append(kp_obs_enc_h)
        enc_last_state_list.append(kp_obs_enc_last_state)

        # encode person scene
        obs_personscene = batch['ctx'][:, -1] # B 4 (T) H W -> B (T) H W
        if len(obs_personscene.shape) == 4:
            obs_personscene = obs_personscene[:,-1]
        try:
            obs_personscene = F.one_hot(obs_personscene.long(), self.n_seg)  # B H W n_seg
        except:
            import pdb; pdb.set_trace()
        obs_personscene = obs_personscene.float()  # B H W n_seg
        obs_personscene = self.conv3(self.conv2(obs_personscene)).mean(1).mean(1)  # B C
        obs_personscene = obs_personscene.unsqueeze(1).repeat(1, T, 1)  # B T C
        personscene_obs_enc_h, personscene_obs_enc_last_state = \
            self.enc_personscene(obs_personscene)
        enc_h_list.append(personscene_obs_enc_h)
        enc_last_state_list.append(personscene_obs_enc_last_state)

        # encode person obj
        other_geo = batch['social'][:,:,:,:4].permute(0,2,1,3)  # B K T 4 --> B T K 4
        other_cls = batch['social'][:,:,:,-1].unsqueeze(-1).permute(0,2,1,3)  # B K T 1 --> B T K 1
        obs_other_boxes_geo_features = self.other_box_geo_emb(other_geo)
        obs_other_boxes_class_features = self.other_box_class_emb(other_cls)
        obs_other_boxes_features = torch.cat(
            [obs_other_boxes_geo_features, obs_other_boxes_class_features],
            dim=3)
        # cosine simi
        obs_other_boxes_geo_features = F.normalize(obs_other_boxes_geo_features, p=2, dim=-1)
        obs_other_boxes_class_features = F.normalize(
            obs_other_boxes_class_features, p=2, dim=-1)
        # [N, T,K]
        other_attention = torch.multiply(
            obs_other_boxes_geo_features, obs_other_boxes_class_features).sum(3)

        other_attention = torch.softmax(other_attention,dim=-1)  # [N, T,K]
        # [N, obs_len, K, 1] * [N, obs_len, K, feat_dim]
        # -> [N, obs_len, feat_dim]
        other_box_features_attended = (torch.unsqueeze(
            other_attention, -1) * obs_other_boxes_features).mean(dim=2)  # sum K

        other_obs_enc_h, other_obs_enc_last_state = self.enc_other(other_box_features_attended)
        enc_h_list.append(other_obs_enc_h)
        enc_last_state_list.append(other_obs_enc_last_state)

        obs_enc_h = torch.stack(enc_h_list, dim=1)
        obs_enc_last_state = concat_states(enc_last_state_list, dim=1).h  # B 5*d
        traj_obs_last = batch['traj'][:, -1]
        pred_length = self.pred_len
        traj_pred_out = self.decode(traj_obs_last, traj_obs_enc_last_state,
                                         obs_enc_h, pred_length, self.dec_cell_traj)
        out = {'pred_traj': traj_pred_out,
               'cls_logits': {}}
        for k in self.action_sets:
            out['cls_logits'][k] = self.future_act[k](obs_enc_last_state)
        
        return out
