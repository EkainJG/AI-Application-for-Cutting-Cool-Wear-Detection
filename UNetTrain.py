import csv
import cv2
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import glob
import numpy as np
import random



# RUTAS

train_images = "C:\\Ekain\\data\\22022026_dataset\\images_data"  # cargar imágenes de entrenamiento
train_masks = "C:\\Ekain\\data\\22022026_dataset\\Train_Msk"   # cargar máscaras de entrenamiento  
val_images = "C:\\Ekain\\data\\22022026_dataset\\img_val"   # cargar imágenes de validación
val_masks = "C:\\Ekain\\data\\22022026_dataset\\Val_Msk"  # cargar máscaras de validación      


#CONFIGURACION 

ENCODER_NAME = "efficientnet-b0"  # encoder preentrenado
ENCODER_WEIGHTS = "imagenet"  # pesos preentrenados en ImageNet
BATCH_SIZE = 6
NUM_WORKERS = 4
NUM_EPOCHS_FROZEN = 12
NUM_EPOCHS_UNFROZEN = 200
PATIENCE = 16
FROZEN_LR = 1e-4
ENCODER_LR = 1e-5
DECODER_LR = 1e-4
SEGMENTATION_HEAD_LR = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 3          # encoder LR warmup after unfreeze
CSV_FILE = "two_phase2.csv"
BEST_CKPT = "best_two_phase2.pth"
FINAL_CKPT = "final_two_phase2.pth"

# SEEDS

# Se establece un semilla para la reproducibilidad del entrenamiento
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#
# TRANSFORMACIONES
#


# data augmentation y formto para el entrenamiento
train_transform = A.Compose([
    A.RandomCrop(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
        A.RandomGamma(gamma_limit=(80, 120)),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8)),
    ], p=0.5),

    A.OneOf([
        A.GaussNoise(std_range=(0.05, 0.1), mean_range=(0.0, 0.0)),
        A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.1, 2.0)),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
    ], p=0.4),

    A.OneOf([
        A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-15, 15), val_shift_limit=(-10, 10)),
        A.RGBShift(r_shift_limit=(-10, 10), g_shift_limit=(-10, 10), b_shift_limit=(-10, 10)),
    ], p=0.3),

    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])


# 
# DATASET
# 

class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, masks, transform=None, ensure_positive=False, max_empty_ratio=0.3, max_retries=5):
        self.images = images
        self.masks = masks

        self.transform = transform
        self.ensure_positive = ensure_positive
        self.max_empty_ratio = max_empty_ratio
        self.max_retries = max_retries

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            retries = self.max_retries if self.ensure_positive else 1
            for attempt in range(retries):
                augmented = self.transform(image=image, mask=mask)
                aug_mask = augmented["mask"]
                has_wear = (aug_mask.sum() > 0).item()
                accept_empty = np.random.random() < self.max_empty_ratio
                if (has_wear or accept_empty) or not self.ensure_positive or attempt == retries - 1:
                    image = augmented["image"]
                    mask = aug_mask.unsqueeze(0)
                    break
        return image, mask


# 
# FUNCION DE PERDIDA
# 
dice_loss = smp.losses.DiceLoss(mode='binary')
bceloss = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(outputs, targets):
    return dice_loss(outputs, targets) + bceloss(outputs, targets)


# 
# VALIDACION
# 
def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    dice_list = []
    iou_list = []

    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images = val_images.to(device)
            val_masks = val_masks.to(device)

            
            val_outputs = model(val_images)
            v_loss = loss_fn(val_outputs, val_masks)



            val_loss += v_loss.item()

            pred_masks = (torch.sigmoid(val_outputs) > 0.5).long()
            truth_masks = val_masks.long()

            tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, truth_masks, mode="binary")
            iou_list.append(smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction="none").mean().item())
            dice_list.append(smp.metrics.functional.f1_score(tp, fp, fn, tn, reduction="none").mean().item())

    val_loss /= len(val_loader)
    dice_mean = np.mean(dice_list)
    iou_mean = np.mean(iou_list)
    return val_loss, dice_mean, iou_mean


# 
# MAIN
# 
def main():
    # ── PATHS ──
    image_paths = sorted(glob.glob(os.path.join(train_images, "*.*")))
    mask_paths = sorted(glob.glob(os.path.join(train_masks, "*.*")))
    val_image_paths = sorted(glob.glob(os.path.join(val_images, "*.*")))
    val_mask_paths = sorted(glob.glob(os.path.join(val_masks, "*.*")))

    
    # ── Modelo ──
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ── Congelar encoder ──
    for param in model.encoder.parameters():
        param.requires_grad = False

    # ── Datasets y Loaders ──
    train_dataset = Dataset(image_paths, mask_paths, transform=train_transform, ensure_positive=True)
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, persistent_workers=True, pin_memory=True, drop_last=False
        
    )

    val_dataset = Dataset(val_image_paths, val_mask_paths, transform=val_transform, ensure_positive=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=NUM_WORKERS,persistent_workers=True, pin_memory=True
        
    )

    # ── Optimizer, Scheduler, Scaler ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FROZEN_LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    # Configuración
    num_epochs_frozen = NUM_EPOCHS_FROZEN
    num_epochs_unfrozen = NUM_EPOCHS_UNFROZEN
    patience = PATIENCE
    best_val_metric = -1.0
    epochs_no_improve = 0
    global_epoch = 0

    # 
    # Fase 1 - Entrenamiento con encoder congelado
    # 
    for epoch in range(num_epochs_frozen):
        global_epoch += 1
        model.train()
        model.encoder.eval()
        epoch_loss = 0

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            
            epoch_loss += loss.item()

        epoch_mean_loss = epoch_loss / len(loader)
        print(f"[Frozen] Epoch {global_epoch} - Loss: {epoch_mean_loss:.4f}")

        #  Validacion
        val_loss, dice_mean, iou_mean = validate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(dice_mean)

        # ── CSV logging ──
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                global_epoch,
                f"{dice_mean:.6f}".replace('.', ','),
                f"{iou_mean:.6f}".replace('.', ','),
                f"{val_loss:.6f}".replace('.', ','),
                f"{epoch_mean_loss:.6f}".replace('.', ','),
            ])

        # ── Checkpointing ──
        if dice_mean > best_val_metric:
            best_val_metric = dice_mean
            torch.save(model.state_dict(), BEST_CKPT)
            epochs_no_improve = 0
            print(f"✓ Nuevo mejor modelo: {dice_mean:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping (frozen) - Best dice: {best_val_metric:.4f}")
            break

        print(f"LR: {current_lr:.6f} - Vdice: {dice_mean:.4f} - Viou: {iou_mean:.4f} - Vloss: {val_loss:.4f}")

    # 
    # Fase 2 - Descongelar encoder y continuar entrenamiento
    # 
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(),          'lr': ENCODER_LR},
        {'params': model.decoder.parameters(),          'lr': DECODER_LR},
        {'params': model.segmentation_head.parameters(),'lr': SEGMENTATION_HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5, min_lr=1e-6)
    epochs_no_improve = 0
    

    for epoch in range(num_epochs_unfrozen):
        global_epoch += 1
        model.train()
        epoch_loss = 0

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

        
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            

            epoch_loss += loss.item()

        epoch_mean_loss = epoch_loss / len(loader)
        print(f"[Unfrozen] Epoch {global_epoch} - Loss: {epoch_mean_loss:.4f}")

        # Validacion
        val_loss, dice_mean, iou_mean = validate(model, val_loader, device)
        encoder_lr = optimizer.param_groups[0]['lr']
        decoder_lr = optimizer.param_groups[1]['lr']
        scheduler.step(dice_mean)

        # Guardar métricas en CSV
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow([
                global_epoch,
                f"{dice_mean:.6f}".replace('.', ','),
                f"{iou_mean:.6f}".replace('.', ','),
                f"{val_loss:.6f}".replace('.', ','),
                f"{epoch_mean_loss:.6f}".replace('.', ','),
            ])

        print(f"LRe: {encoder_lr:.6f} - LRd: {decoder_lr:.6f} - Vdice: {dice_mean:.4f} - Viou: {iou_mean:.4f} - Vloss: {val_loss:.4f}")

        # ── Guardar mejor modelo basado en métrica de validación (dice)
        if dice_mean > best_val_metric:
            best_val_metric = dice_mean
            torch.save(model.state_dict(), BEST_CKPT)
            epochs_no_improve = 0
            print(f"✓ Nuevo mejor modelo: {dice_mean:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping (unfrozen) - Best dice: {best_val_metric:.4f}")
            break

    torch.save(model.state_dict(), FINAL_CKPT)
    print("Done!")



if __name__ == "__main__":
    main()