import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import nibabel as nib


def dice_loss(pred, target, eps=1e-6):
    pred = torch.softmax(pred, dim=1)
    target_1hot = F.one_hot(target, pred.shape[1]).permute(0,4,1,2,3)

    intersection = (pred * target_1hot).sum(dim=(0,2,3,4))
    union = pred.sum(dim=(0,2,3,4)) + target_1hot.sum(dim=(0,2,3,4))

    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()

def combined_loss(pred, target):
    # ce = F.cross_entropy(pred, target)
    dl = dice_loss(pred, target)
    # return ce + dl
    return  dl


def calculate_dice(preds, targets, num_classes=4):
    """
    preds: Output của model (Logits) [Batch, C, D, H, W]
    targets: Ground Truth [Batch, D, H, W]
    """
    # Chuyển logits thành xác suất rồi lấy class có xác suất cao nhất
    preds = torch.argmax(torch.softmax(preds, dim=1), dim=1) # [B, D, H, W]
    
    dice_per_class = []
    # Bỏ qua class 0 (Background) vì nó chiếm đa số, tính vào sẽ làm ảo chỉ số
    for c in range(1, num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = (pred_c & target_c).float().sum()
        union = pred_c.float().sum() + target_c.float().sum()
        
        if union == 0:
            dice = 1.0 # Cả 2 đều không có class này => dự đoán đúng
        else:
            dice = (2.0 * intersection) / (union + 1e-8) # +epsilon để tránh chia cho 0
        dice_per_class.append(dice.item())
        
    return sum(dice_per_class) / len(dice_per_class) # Trả về Dice trung bình của 3 class (1, 2, 3)

