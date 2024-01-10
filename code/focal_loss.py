import numpy as np
import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        
        # inputs가 numpy.ndarray인 경우에만 변환
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)
        
        # targets가 numpy.ndarray인 경우에만 변환
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        
        inputs = inputs.float()  # torch.float32로 변환
        
        #print("Inputs shape:", inputs.shape)  # 확인을 위한 출력 추가
        #print("Tatgets shape:", targets.shape)  
        #print("Inputs:", inputs)
        #print("Tatgets:", targets)  
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return torch.mean(focal_loss) if self.reduction == 'mean' else torch.sum(focal_loss)