import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0

        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


"专为解决非平衡数据设计的，特别适合需要精细控制 FP 和 FN 权衡的场景。"
"""提高 Recall: 调大 beta (例如 0.7)，同时调小 alpha。这会重罚漏报行为。
提高 Precision: 调大 alpha (例如 0.7)，同时调小 beta。这会重罚误报行为。"""
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=2.0, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  

        FocalTversky = (1 - Tversky)**self.gamma
        
        return FocalTversky



class DiceFocalLoss(nn.Module):
    def __init__(self, pos_weight_val, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = torch.tensor([pos_weight_val]).to(DEVICE)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=self.weight, reduction='none')
        
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        logits = torch.clamp(logits, min=-100, max=100) 
        probas = torch.sigmoid(logits)
        
        p_t = targets * probas + (1 - targets) * (1 - probas)
        p_t = torch.clamp(p_t, min=1e-7, max=1.0) 
        focal_loss = (self.alpha * (1 - p_t)**self.gamma * bce_loss).mean()
        
        smooth = 1e-5
        probs_flat = probas.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (probs_flat.sum() + targets_flat.sum() + smooth)
        
        return 0.5 * focal_loss + 0.5 * dice_loss

class DiceFocalLoss_optimize(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, weight_dice=0.4, weight_focal=0.7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        inputs = probs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy_with_logits(logits.view(-1), targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_loss = (alpha_t * (1 - p_t) ** self.gamma * bce).mean()

        smooth = 1e-5
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return self.weight_dice * dice_loss + self.weight_focal * focal_loss


class BoundaryAwareFocalLoss(nn.Module):
    def __init__(self, alpha=0.90, gamma=2.0, boundary_weight=20.0, internal_weight=5.0, codon_len=3):
        super(BoundaryAwareFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.internal_weight = internal_weight
        self.codon_len = codon_len
        #print(f"[Loss Init] Alpha={alpha}, Gamma={gamma}, Boundary W={boundary_weight}, Internal W={internal_weight}")

    def generate_weight_map(self, targets):
        weight_map = torch.ones_like(targets, dtype=torch.float32)

        weight_map = torch.where(targets > 0.5, 
                                 torch.tensor(self.internal_weight, device=targets.device), 
                                 weight_map)

        padded_targets = F.pad(targets, (1, 1), mode='constant', value=0)
        diff = padded_targets[:, 1:] - padded_targets[:, :-1]
        diff = diff[:, :-1] 

        starts = (diff == 1).float()
        ends = (diff == -1).float()

        start_mask = torch.zeros_like(targets)
        end_mask = torch.zeros_like(targets)
        
        for k in range(self.codon_len):
            s_s = torch.roll(starts, shifts=k, dims=1)
            s_s[:, :k] = 0
            start_mask = torch.max(start_mask, s_s)

            e_s = torch.roll(ends, shifts=-(k+1), dims=1)
            e_s[:, -(k+1):] = 0
            end_mask = torch.max(end_mask, e_s)
            
        boundary_mask = torch.max(start_mask, end_mask)
        
        # 应用边界权重
        weight_map = torch.where(boundary_mask > 0.5, 
                                 torch.tensor(self.boundary_weight, device=targets.device), 
                                 weight_map)
        
        return weight_map

    def forward(self, logits, targets):
        logits = logits.float() 
        targets = targets.float()

        # 1. 维度对齐
        if logits.dim() == 3: logits = logits.squeeze(1)
        if targets.dim() == 3: targets = targets.squeeze(1)

        # === [核心修复 2] 增加 Logits 截断保护 ===
        # 防止 logits 过大导致 sigmoid 后变成绝对的 0 或 1，进而导致 log(0)=inf
        logits = torch.clamp(logits, min=-100, max=100)

        # 2. 生成权重图 (No Grad)
        with torch.no_grad():
            pixel_weights = self.generate_weight_map(targets)

        # 3. 计算 BCE (使用 stable 的 with_logits)
        # reduction='none' 保留每个像素的 loss 用于后续加权
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 4. 计算 Focal Term
        pt = torch.exp(-bce_loss) # pt 是预测正确的概率
        
        # 计算 Alpha 平衡
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
            
        # (1 - pt) ** gamma * bce_loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * bce_loss
        
        # 5. 应用空间权重 (Boundary/Internal)
        weighted_loss = focal_loss * pixel_weights
        
        # 6. 最终平均
        # 增加一个极小值防止除以0 (虽然 mean 不太可能)
        loss = weighted_loss.mean()
        
        return loss
    
class ComprehensiveBoundaryLoss(nn.Module):
    def __init__(self, 
                 # Focal / Boundary 参数
                 alpha=0.90, 
                 gamma=2.0, 
                 boundary_weight=20.0, 
                 internal_weight=5.0, 
                 codon_len=3,
                 # 组合权重参数
                 lambda_boundary=1.0,  # 边界感知 Focal 的权重
                 lambda_dice=0.5       # Dice Loss 的权重
                 ):
        """
        ComprehensiveBoundaryLoss: 综合了边界感知与全局形状优化的终极损失函数。
        
        参数详解:
            alpha (float): Focal Loss 正样本平衡参数。建议 0.90 (强力提升 Recall)。
            gamma (float): Focal Loss 聚焦参数。建议 2.0。
            boundary_weight (float): 边界区域(ATG/Stop)的惩罚倍数。建议 20.0。
            internal_weight (float): sORF 内部区域的惩罚倍数。建议 5.0。
            lambda_boundary (float): BoundaryAwareFocalLoss 在总 Loss 中的占比权重。
            lambda_dice (float): Dice Loss 在总 Loss 中的占比权重。
        """
        super(ComprehensiveBoundaryLoss, self).__init__()
        # 核心参数
        self.alpha = alpha
        self.gamma = gamma
        self.boundary_weight = boundary_weight
        self.internal_weight = internal_weight
        self.codon_len = codon_len
        
        # 组合权重
        self.lambda_boundary = lambda_boundary
        self.lambda_dice = lambda_dice
        
        print(f"[Loss Init] Comprehensive Mode: Boundary(x{lambda_boundary}) + Dice(x{lambda_dice})")
        print(f"            Details: Alpha={alpha}, BoundW={boundary_weight}, InternW={internal_weight}")

    def generate_weight_map(self, targets):
        """
        生成空间权重图: 
        - 背景: 1.0
        - 内部: internal_weight
        - 边界: boundary_weight
        """
        # 1. 基础权重图
        weight_map = torch.ones_like(targets, dtype=torch.float32)
        
        # 2. 内部加权 (target=1 的区域)
        weight_map = torch.where(targets > 0.5, 
                                 torch.tensor(self.internal_weight, device=targets.device), 
                                 weight_map)

        # 3. 边界加权 (使用差分法找边缘)
        # Pad 以处理序列两端
        padded_targets = F.pad(targets, (1, 1), mode='constant', value=0)
        diff = padded_targets[:, 1:] - padded_targets[:, :-1]
        diff = diff[:, :-1] # 还原长度

        starts = (diff == 1).float()  # 0->1
        ends = (diff == -1).float()   # 1->0

        # 膨胀边界 (Dilation)
        start_mask = torch.zeros_like(targets)
        end_mask = torch.zeros_like(targets)
        
        for k in range(self.codon_len):
            # Start: 向右延伸覆盖 ATG
            s_s = torch.roll(starts, shifts=k, dims=1)
            s_s[:, :k] = 0
            start_mask = torch.max(start_mask, s_s)
            
            # End: 向左延伸覆盖 Stop Codon
            e_s = torch.roll(ends, shifts=-(k+1), dims=1)
            e_s[:, -(k+1):] = 0
            end_mask = torch.max(end_mask, e_s)
            
        boundary_mask = torch.max(start_mask, end_mask)
        
        # 应用边界权重 (覆盖内部权重)
        weight_map = torch.where(boundary_mask > 0.5, 
                                 torch.tensor(self.boundary_weight, device=targets.device), 
                                 weight_map)
        return weight_map

    def forward(self, logits, targets):
        """
        logits: 模型输出 [Batch, Length] (未经过 Sigmoid)
        targets: 真实标签 [Batch, Length]
        """
        # === 1. 数值稳定性处理 (A800/AMP 必备) ===
        logits = logits.float()   # 强制转 FP32 防止溢出
        targets = targets.float()
        
        # 维度对齐
        if logits.dim() == 3: logits = logits.squeeze(1)
        if targets.dim() == 3: targets = targets.squeeze(1)

        # Logits 截断保护 (防止 sigmoid 后变为绝对 0/1 导致 log 炸裂)
        logits = torch.clamp(logits, min=-100, max=100)
        
        # === 2. 计算 Boundary Aware Focal Loss ===
        # A. 生成权重图 (无需梯度)
        with torch.no_grad():
            pixel_weights = self.generate_weight_map(targets)
            
        # B. 计算基础 BCE (保留 pixel-wise loss)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # C. Alpha 平衡
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
            
        # D. 结合 Focal + Spatial Weights
        # Loss = Weight_Map * Alpha * (1-pt)^gamma * BCE
        focal_term = pixel_weights * alpha_t * (1 - pt) ** self.gamma * bce_loss
        boundary_loss = focal_term.mean()
        
        # === 3. 计算 Dice Loss ===
        probs = torch.sigmoid(logits)
        
        # 展平计算全局 Dice
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        smooth = 1e-6
        intersection = (probs_flat * targets_flat).sum()
        union = probs_flat.sum() + targets_flat.sum()
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score
        
        # === 4. 综合输出 ===
        total_loss = self.lambda_boundary * boundary_loss + self.lambda_dice * dice_loss
        
        return total_loss
    
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=20.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # [核心] 这里把正样本的 Loss 放大 20 倍
        self.pos_weight = torch.tensor([pos_weight]).cuda() 

    def forward(self, inputs, targets):
        # 1. 计算带权重的 BCE
        # pos_weight 参数会让正样本的 Loss 直接乘以 20
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        
        pt = torch.exp(-bce_loss) # 简化计算
        
        # 2. Focal Term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()