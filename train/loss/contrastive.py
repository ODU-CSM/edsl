import torch
import torch.nn as nn
import torch.distributed as dist
from pytorch_metric_learning import losses
import torch.nn.functional as F


#class NT_Xent(nn.Module):
class ConLoss(nn.Module):
    def __init__(self, batch_size=64, temperature=0.01, world_size=1):
        super(ConLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.supConLoss = losses.SupConLoss(temperature)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j, mode=None):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        if mode == 'cont':
            N = 2 * self.batch_size * self.world_size

            z = torch.cat((z_i, z_j), dim=0)
            if self.world_size > 1:
                z = torch.cat(GatherLayer.apply(z), dim=0)

            sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

            sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
            sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

            # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            negative_samples = sim[self.mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
            return loss
        elif mode == 'selfCon':
            feats = torch.cat([z_i[:, 0, :], z_i[:, 1, :]])
            # Calculate cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
            # Mask out cosine similarity to itself
            self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
            cos_sim.masked_fill_(self_mask, -9e15)
            # Find positive example -> batch_size//2 away from the original example
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
            # InfoNCE loss
            cos_sim = cos_sim / self.temperature
            nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
            loss = nll.mean()
            return loss
        elif mode == 'semiCon':
            features = z_i
            labels = z_j
            device = features.device
            batch_size = features.shape[0]
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

            contrast_count = features.shape[1]
            contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
            anchor_feature = contrast_feature
            anchor_count = contrast_count

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(anchor_feature, contrast_feature.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # tile mask
            mask = mask.repeat(anchor_count, contrast_count)
            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            # nomenator - denominator
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

            # compute mean of log-likelihood over positive
            # sum over P
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - mean_log_prob_pos
            # outer sum + division over P
            loss = loss.view(anchor_count, batch_size).mean()
            return loss
        elif mode == 'supCon':
            labels = torch.cat([z_j.squeeze(), z_j.squeeze()])
            features = torch.cat([z_i[:, 0, :], z_i[:, 1, :]])
            loss = self.supConLoss(features, labels)
            return loss
        elif mode == 'supCon2':
            labels = torch.cat([z_j.squeeze(), z_j.squeeze()])
            features = torch.cat([z_i[:, 0, :], z_i[:, 1, :]])
            features_normalized = torch.nn.functional.normalize(features, p=2, dim=1)
            # Compute logits
            logits = torch.div(
                torch.matmul(
                    features_normalized, torch.transpose(features_normalized, 0, 1)
                ),
                self.temperature,
            )
            return losses.NTXentLoss(temperature=self.temperature)(logits, torch.squeeze(labels))
        else:
            prediction = z_i
            target = z_j
            loss = F.binary_cross_entropy_with_logits(prediction, target)
            return loss
            '''
            loss = nn.functional.binary_cross_entropy(prediction, target, reduction='none')
            sumOfLoss = torch.sum(loss)
            nEffTerms = torch.tensor(torch.numel(target))
            if nEffTerms == 0:
                aveOfLoss = torch.tensor(0)
            else:
                aveOfLoss = torch.div(sumOfLoss, nEffTerms)
            return aveOfLoss
            '''

    def setReprLoss(self, reprLoss):
        print()
