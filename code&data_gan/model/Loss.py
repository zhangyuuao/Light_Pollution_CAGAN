import torch.nn as nn
import torch

class L1_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, output):
        return torch.norm(input - output, p = 1)

class L2_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, output):
        return torch.norm(input - output, p = 2)
    
class ELBO_Loss(nn.Module):
    def __init__(self, kld_weight: float = 0.1):
        super().__init__()
        self.l1loss = L1_loss()
        self.kld_weight = kld_weight
        
    def forward(self, recon_x, x, mean, logstd):
        L1loss = self.l1loss(recon_x, x)
        kld = -0.5 * (torch.sum(1 + logstd - torch.exp(logstd) - mean ** 2))
        return L1loss + self.kld_weight * kld
