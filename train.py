from .utils import dice_loss, combined_loss, calculate_dice
from unet import UNet3D
from cnn3d import CNN3DModel
from .dataset import BratsDataset
import torch




def trainer(model: torch.nn.Module(), model_type: str, train_loader, val_loader):
    for epoch in range(8):
        model.train()
        losses = []
        for img, mask in tqdm(train_loader):
            img = img.cuda()
            mask = mask.cuda()

            pred = model(img)
            loss = combined_loss(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}: Loss = {sum(losses)/len(losses):.4f}")
        if epoch%3 == 1:
            model.eval()
            val_losses = []
            val_dices = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/20 [Valid]")
                for img, mask in val_pbar:
                    img = img.cuda()
                    mask = mask.cuda()

                    pred = model(img)
                    
                    # 1. Tính Loss
                    loss = combined_loss(pred, mask)
                    val_losses.append(loss.item())
                    
                    # 2. Tính Dice Score (Metric đánh giá thực tế)
                    dice = calculate_dice(pred, mask, num_classes=4)
                    val_dices.append(dice)
                    
                    val_pbar.set_postfix({'val_loss': loss.item(), 'dice': dice})

            avg_val_loss = sum(val_losses) / len(val_losses)
            avg_val_dice = sum(val_dices) / len(val_dices)

            # ================= LOGGING & SAVE =================
            print(f"\nEND EPOCH {epoch+1}:")
            # print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Valid Loss: {avg_val_loss:.4f} | Valid Dice: {avg_val_dice:.4f}")

            # Chỉ lưu model nếu Dice score cải thiện
            if avg_val_dice > best_dice:
                print(f"  >>> Model Improved (Dice: {best_dice:.4f} -> {avg_val_dice:.4f}). Saving...")
                torch.save(model.state_dict(), f"/dungnq/best_{model_type}.pth")
                best_dice = avg_val_dice
            
            print("-" * 50)


if __name__ == '__main__':
    import torch
    from torch.utils.data import DataLoader, random_split
    from tqdm import tqdm
    import nibabel as nib

    # Giả sử bạn đã import BratsDataset và UNet3D
    # from dataset import BratsDataset
    # from model import UNet3D

    # 1. Khởi tạo Dataset gốc
    full_ds = BratsDataset("BraTS2021_Training_Data")

    # 2. Tính toán kích thước cho từng tập (Tỉ lệ 8:1:1)
    total_size = len(full_ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size # Lấy phần còn lại để đảm bảo tổng không bị lệch do làm tròn

    print(f"Tổng số mẫu: {total_size}")
    print(f"Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # 3. Thực hiện chia ngẫu nhiên (Dùng generator để cố định seed giúp kết quả lặp lại được)
    generator = torch.Generator().manual_seed(42) 
    train_set, val_set, test_set = random_split(full_ds, [train_size, val_size, test_size], generator=generator)

    # 4. Tạo DataLoader cho từng tập
    # Lưu ý: batch_size=16 cho 3D là RẤT LỚN, dễ bị tràn VRAM (OOM). 
    # Với 3D UNet thường chỉ để batch_size=1 hoặc 2 tùy GPU.
    batch_size = 16

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4) # Val không cần shuffle
    test_loader  = DataLoader(test_set,  batch_size=1,          shuffle=False, num_workers=4) # Test thường batch=1 để đánh giá từng ca

    # 5. Khởi tạo Model và Optimizer
    model = UNet3D(n_channels=1, n_classes=4).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train(model,'unet3d_brats', train_loader, val_loader)
