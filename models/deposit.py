import math

import torch
import torch.nn as nn
import numpy as np
from .diffusion_util import diff_CSDI

deposit_config = {
    'train':
        {
            'epochs': 100,
            'batch_size': 32,
            'batch_size_test': 32,
            'lr': 1.0e-3
        },
    'diffusion':
        {
            'layers': 4,
            'channels': 64,
            'nheads': 8,
            'diffusion_embedding_dim': 128,
            'beta_start': 0.0001,
            'beta_end': 0.5,
            'num_steps': 50,
            'schedule': "cosine"
        },
    'model':
        {
            'is_unconditional': 0,
            'timeemb': 128,
            'featureemb': 16
        }
}

class Deposit(nn.Module):
    def __init__(self, config, device, target_dim=34, w_mm_cond=0, mm_cond_dim=512, modality='sklt'):
        super().__init__()
        self.device = device
        self.target_dim = target_dim  # joints * n_coords or 4(or 5 or 1)
        self.w_mm_cond = w_mm_cond
        self.mm_cond_dim = mm_cond_dim
        self.modality = modality

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for conditional mask
        if self.w_mm_cond:
            self.emb_total_dim += self.mm_cond_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )
        elif config_diff["schedule"] == "cosine":
            self.beta = self.betas_for_alpha_bar(
                self.num_steps,
                lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.5):
        # """
        # Create a beta schedule that discretizes the given alpha_t_bar function,
        # which defines the cumulative product of (1-beta) over time from t = [0,1].
        # :param num_diffusion_timesteps: the number of betas to produce.
        # :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
        #                   produces the cumulative product of (1-beta) up to that
        #                   part of the diffusion process.
        # :param max_beta: the maximum beta to use; use values lower than 1 to
        #                  prevent singularities.
        # """
        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def time_embedding(self, pos, d_model=128):
        '''
        pos: B,T
        '''
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(pos.device)  # B T d
        position = pos.unsqueeze(2)  # B T 1
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(pos.device) / d_model
        )  # d/2,
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, observed_tp, cond_mask, mm_cond=None):
        '''
        observed_tp: B,T
        cond_mask: B,C,T
        mm_cond: B,mm_cond_dim
        '''
        try:
            B, K, L = cond_mask.shape  # B 54 50
        except:
            import pdb; pdb.set_trace()

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,t_emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)  # (B,L,K,t_emb)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(observed_tp.device)
        )  # (K,feat_emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        try:
            side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        except:
            import pdb; pdb.set_trace()
        side_info = side_info.permute(0, 3, 2, 1)  # (B,t_emb+f_emb,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)  # B,t_emb+f_emb+1,K,L
        
        if self.w_mm_cond:
            mm_cond = mm_cond.unsqueeze(2).unsqueeze(3).expand(-1, -1, K, L)
            side_info = torch.cat([side_info, mm_cond], dim=1)  # B,t_emb+f_emb+1+cond_emb,K,L

        return side_info

    def calc_loss_valid(
            self, observed_data, cond_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
            self, observed_data, cond_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long()
        else:
            t = torch.randint(0, self.num_steps, [B])
        current_alpha = self.alpha_torch[t].to(observed_data.device)  # (B,1,1)
        noise = torch.randn_like(observed_data).to(observed_data.device)  # normal gaussian
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t.to(total_input.device))  # (B,K,L)

        target_mask = 1 - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        '''
        '''
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            # add noise to absent parts
            cond_obs = (cond_mask * observed_data).unsqueeze(1)  # (B,1,K,L)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(observed_data.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(observed_data.device))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                                    (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                            ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = (current_sample * (1 - cond_mask) + observed_data * cond_mask).detach()
        return imputed_samples

    def forward(self, batch, n_samples=5, mm_cond=None, is_train=1):
        '''
        batch: (inputs, target)
        '''
        (
            observed_data,  # B C T
            observed_tp,
            gt_mask
        ) = self.process_data(batch)
        cond_mask = gt_mask  # B C T
        side_info = self.get_side_info(observed_tp=observed_tp, 
                                       cond_mask=cond_mask, 
                                       mm_cond=mm_cond)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        loss = loss_func(observed_data, cond_mask, side_info, is_train)
        samples = None
        # prediction
        with torch.no_grad():
            # B K nj*ndim T
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return {'loss': loss,
                'pred': samples}

    def evaluate(self, batch, n_samples, mm_cond=None):
        (
            observed_data,  # B obs+pred 2*nj
            observed_tp,  # B obs+pred,
            gt_mask  # B obs+pred 2*nj
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = 1 - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)
        return samples, observed_data, target_mask, observed_tp

    # def _process_data(self, batch):
        
    #     pose = batch["pose"].to(self.device).float()
    #     tp = batch["timepoints"].to(self.device).float()
    #     mask = batch["mask"].to(self.device).float()

    #     pose = pose.permute(0, 2, 1)  # B C T
    #     mask = mask.permute(0, 2, 1)  # B C T

    #     return (
    #         pose,
    #         tp,
    #         mask
    #     )
    
    def process_data(self, batch):
        inputs, targets = batch
        if self.modality == 'sklt':
            input_seq = inputs['sklt']  # B 2 obslen nj
            gt_seq = targets['pred_sklt']  # B 2 predlen nj
            batch_size = input_seq.size(0)
            n_dim = input_seq.size(1)
            obslen = input_seq.size(2)
            predlen = gt_seq.size(2)
            nj = input_seq.size(3)
            seq = torch.concat([input_seq, gt_seq], 2)  # B 2 obslen+predlen nj
            seq = seq.permute(0,1,3,2).reshape(batch_size,-1,obslen+predlen)  # B 2*nj obslen+predlen
            # seq = seq.permute(0,2,1,3).reshape(batch_size, obslen+predlen, -1)  # B obslen+predlen 2*nj
        elif self.modality == 'traj':
            input_seq = inputs['traj']  # B obslen 4
            gt_seq = targets['pred_traj']  # B predlen 4
            batch_size = input_seq.size(0)
            obslen = input_seq.size(1)
            predlen = gt_seq.size(1)
            seq = torch.concat([input_seq, gt_seq], 1)  # B obslen+predlen 4
            seq = seq.permute(0,2,1)  # B obslen+predlen 4 -> B 4 obslen+predlen

        mask = torch.zeros_like(seq).to(seq.device).float()  # B C obslen+predlen
        mask[:, :, :obslen] = 1
        tp = torch.arange(obslen+predlen).unsqueeze(0).repeat(batch_size,1).to(seq.device).float()
        return (
            seq, # B 2*nj obslen+predlen
            tp,  # B obslen+predlen
            mask # B 2*nj obslen+predlen
        )