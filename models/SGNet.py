import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import argparse


def parse_sgnet_args():
    parser = argparse.ArgumentParser(description='sgnet args')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--shuffle', default=True, type=bool)

    parser.add_argument('--dataset', default='PIE', type=str)
    parser.add_argument('--lr', default=5e-04, type=float)
    parser.add_argument('--data_root', default='data/PIE', type=str)
    parser.add_argument('--model', default='SGNet_CVAE', type=str)
    parser.add_argument('--bbox_type', default='cxcywh', type=str)
    parser.add_argument('--normalize', default='zero-one', type=str)
    parser.add_argument('--hidden_size', default=512, type=int)
    parser.add_argument('--enc_steps', default=15, type=int)
    parser.add_argument('--dec_steps', default=45, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    parser.add_argument('--nu', default=0.0, type=float)
    parser.add_argument('--sigma', default=1.5, type=float)
    parser.add_argument('--FPS', default=30, type=int)
    parser.add_argument('--min_bbox', default=[0,0,0,0], type=list)
    parser.add_argument('--max_bbox', default=[1920, 1080, 1920, 1080], type=list)
    parser.add_argument('--K', default=20, type=int)
    parser.add_argument('--DEC_WITH_Z', default=True, type=bool)
    parser.add_argument('--LATENT_DIM', default=32, type=int)
    parser.add_argument('--pred_dim', default=4, type=int)
    parser.add_argument('--input_dim', default=4, type=int)
    return parser.parse_args()

def parse_sgnet_args2():
    return sgnet_args()

class sgnet_args:
    def __init__(self) -> None:
        self.checkpoint = ''
        self.start_epoch = 1
        self.gpu = '0'
        self.num_workers = 8
        self.epochs = 50
        self.batch_size = 128
        self.weight_decay = 5e-04
        self.seed = 1
        self.phases = ['train', 'test']
        self.shuffle = True
        
        self.dataset = 'PIE'
        self.lr = 5e-04
        self.data_root = 'data/PIE'
        self.model = 'SGNet_CVAE'
        self.bbox_type = 'cxcywh'
        self.normalize = 'zero-one'
        self.hidden_size = 512
        self.enc_steps = 15
        self.dec_steps = 45
        self.dropout = 0.0
        self.nu = 0.0
        self.sigma = 1.5
        self.FPS = 30
        self.min_bbox = [0,0,0,0]
        self.max_bbox = [1920, 1080, 1920, 1080]
        self.K = 20
        self.DEC_WITH_Z = True
        self.LATENT_DIM = 32
        self.pred_dim = 4
        self.input_dim = 4


class JAADFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(JAADFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)

        return embedded_box_input

class ETHUCYFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(ETHUCYFeatureExtractor, self).__init__()
        self.embbed_size = args.hidden_size
        self.embed = nn.Sequential(nn.Linear(6, self.embbed_size), 
                                        nn.ReLU()) 


    def forward(self, inputs):
        box_input = inputs

        embedded_box_input= self.embed(box_input)

        return embedded_box_input

class PIEFeatureExtractor(nn.Module):

    def __init__(self, args):
        super(PIEFeatureExtractor, self).__init__()

        self.embbed_size = args.hidden_size
        self.box_embed = nn.Sequential(nn.Linear(4, self.embbed_size), 
                                        nn.ReLU()) 
    def forward(self, inputs):
        box_input = inputs
        embedded_box_input= self.box_embed(box_input)
        return embedded_box_input

_FEATURE_EXTRACTORS = {
    'PIE': PIEFeatureExtractor,
    'JAAD': JAADFeatureExtractor,
    'ETH': ETHUCYFeatureExtractor,
    'HOTEL': ETHUCYFeatureExtractor,
    'UNIV': ETHUCYFeatureExtractor,
    'ZARA1': ETHUCYFeatureExtractor,
    'ZARA2': ETHUCYFeatureExtractor,
}

def build_feature_extractor(args):
    func = _FEATURE_EXTRACTORS[args.dataset]
    return func(args)


def traj2target(obs_bbox: torch.Tensor, 
               pred_bbox: torch.Tensor):
    '''
    obs_bbox: torch.Tensor B,obslen,...
    pred_bbox: torch.Tensor B,predlen,...

    return
        target torch.Tensor B,obslen,predlen,...
    '''
    batch_size = obs_bbox.size(0)
    obslen = obs_bbox.size(1)
    predlen = pred_bbox.size(1)
    bbox_seqs = torch.concat([obs_bbox, pred_bbox], 1)
    target = torch.zeros(batch_size, obslen, predlen, obs_bbox.size(-1)).to(obs_bbox.device)
    for i in range(obslen):
        target[:, i, :, :] = bbox_seqs[:, i+1:i+1+predlen, :] - bbox_seqs[:, i:i+1, :]
    return target

def target2predtraj(target: torch.Tensor,
                obs_bbox: torch.Tensor,):
    '''
    target: torch.tensor B,obslen,predlen,4
    obs_bbox: torch.tensor B,obslen,4
    '''
    obslen = obs_bbox.size(1)
    predlen = target.size(2)
    pred_bbox = obs_bbox[:, -1:, :] + target[:, -1, :, :]
    assert pred_bbox.size(1) == predlen
    return pred_bbox



class SGNet(nn.Module):
    def __init__(self, args):
        super(SGNet, self).__init__()

        self.hidden_size = args.hidden_size
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        if self.dataset in ['JAAD','PIE']:
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                     self.pred_dim),
                                                     nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))  
             
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))


        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)

        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)

    def SGE(self, goal_hidden):
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            # regress goal traj for loss
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def decoder(self, dec_hidden, goal_for_dec):
        # initial trajectory tensor
        dec_traj = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.pred_dim)
        for dec_step in range(self.dec_steps):
            goal_dec_input = dec_hidden.new_zeros(dec_hidden.size(0), self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)#.view(goal_hidden.size(0), self.dec_steps, self.hidden_size//4).sum(1)
            
            
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            dec_traj[:,dec_step,:] = self.regressor(dec_hidden)
        return dec_traj
        
    def encoder(self, traj_input, flow_input=None, start_index = 0):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        for enc_step in range(start_index, self.enc_steps):
            
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            if self.dataset in ['JAAD','PIE', 'ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
                enc_hidden = traj_enc_hidden
            # generate hidden states for goal and decoder 
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)

            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            dec_traj = self.decoder(dec_hidden, goal_for_dec)

            # output 
            all_goal_traj[:,enc_step,:,:] = goal_traj
            all_dec_traj[:,enc_step,:,:] = dec_traj
        
        return all_goal_traj, all_dec_traj
            

    def forward(self, batch, start_index = 0):
        traj_inputs = batch['traj']  # B T 4
        if self.dataset in ['JAAD','PIE']:
            traj_feat = self.feature_extractor(traj_inputs)
            all_goal_traj, all_dec_traj = self.encoder(traj_feat)
            traj_pred = accumulate_traj(traj_inputs, all_dec_traj)
            out = {
                'pred_traj': traj_pred,  # B T 4
                'ori_output': (all_goal_traj, all_dec_traj)  # B obelen predlen 4
            }
            return out
            return all_goal_traj, all_dec_traj
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(traj_inputs[:,start_index:,:])
            traj_feat = traj_input_temp.new_zeros((traj_inputs.size(0), traj_inputs.size(1), traj_input_temp.size(-1)))
            traj_feat[:,start_index:,:] = traj_input_temp
            all_goal_traj, all_dec_traj = self.encoder(traj_feat, None, start_index)
            return all_goal_traj, all_dec_traj


def accumulate_traj(obs_traj: torch.Tensor, 
                    target: torch.Tensor):
    '''
    obs_traj: B obslen 4
    target: B obslen predlen (K) 4
    '''
    obs_traj = obs_traj.unsqueeze(2)  # B obslen 1 4
    if len(target.size()) == 5:  # case of stochastic model
        obs_traj = obs_traj.unsqueeze(3)
    try:
        target += obs_traj
    except:
        import pdb; pdb.set_trace()
    return target[:, -1]  # B predlen ...

def traj_to_sgnet_target(obs_traj: torch.Tensor, 
                         pred_traj: torch.Tensor):
    '''
    obs_traj: B obslen, ...
    pred_traj: B obslen, ...
    '''
    obslen = obs_traj.size(1)
    predlen = obs_traj.size(1)
    seq = torch.concat([obs_traj, pred_traj], dim=1)
    target = []
    for i in range(obslen):
        target.append(
            seq[:, i+1:i+1+predlen] - seq[:, i:i+1]
        )
    target = torch.stack(target, dim=1)  # B obslen predlen ...
    return target



if __name__ == '__main__':
    args = parse_sgnet_args()
    args.enc_steps = 4
    args.dec_steps = 8
    x = torch.rand((2, 4, 4))
    batch = {'traj': x}
    model = SGNet(args)
    out = model(batch)
    print(out['traj_pred'].shape)
