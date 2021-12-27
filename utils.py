import os
import torch
import torchvision
from dataset import LungCTScanDataset
from torch.utils.data import DataLoader
from model import UNET


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = LungCTScanDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = LungCTScanDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def main():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 1
    NUM_WORKERS = 2
    IMAGE_HEIGHT = 256  # 1280 originally
    IMAGE_WIDTH = IMAGE_HEIGHT  # 1918 originally
    PIN_MEMORY = True
    LUNG = "lung"
    PATHOLOGY = "pathology"
    CLASS = LUNG
    VAL_IMG_DIR = "data/" + CLASS + "/val_images/"
    VAL_MASK_DIR = "data/" + CLASS + "/val_masks/"
    SAVE_IMAGE = True

    file_names = [name[:-4] for name in os.listdir(VAL_IMG_DIR)]

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    val_ds = LungCTScanDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transforms,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    # load model
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load("my_checkpoint_" + CLASS + "_" + str(IMAGE_WIDTH) + ".pth.tar"), model)
    num_correct = 0
    num_pixels = 0
    dice_numerator = 0
    dice_denominator = 0
    model.eval()

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_numerator += (2 * (preds * y).sum())
            dice_denominator += (
                    (preds == 1).sum() + (y == 1).sum()
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}")
    print(f"Dice score: {dice_numerator / dice_denominator}")

    if SAVE_IMAGE:
        model.eval()
        for idx, (x, y) in enumerate(val_loader):
            x = x.to(DEVICE)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds > 0.5).float()
            if not os.path.isdir(f"saved_images/{CLASS}/individual_images/{file_names[idx][:9]}"):
                os.mkdir(f"saved_images/{CLASS}/individual_images/{file_names[idx][:9]}")
            torchvision.utils.save_image(
                preds, f"saved_images/{CLASS}/individual_images/{file_names[idx][:9]}/{file_names[idx]}_pred.png"
            )
            # torchvision.utils.save_image(y.unsqueeze(1), f"saved_images/{CLASS}/individual_images/{file_names[idx]}.png")


if __name__ == "__main__":
    main()