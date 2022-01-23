import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

LUNG = "lung"
PATHOLOGY = "pathology"  # consolidation
PARASITIC_INFECTION = "pi"

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
NUM_EPOCHS = 10
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = IMAGE_HEIGHT  # 1918 originally
CLASS = PATHOLOGY
LOAD_MODEL = False

NUM_WORKERS = 2
PIN_MEMORY = True
SAVE_PREDS_AS_IMAGES = True

TRAIN_IMG_DIR = "data/" + CLASS + "/train_images/"
TRAIN_MASK_DIR = "data/" + CLASS + "/train_masks/"
VAL_IMG_DIR = "data/" + CLASS + "/val_images/"
VAL_MASK_DIR = "data/" + CLASS + "/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):  # only one epoch # the for loop iterate n_images/batch_size times # will be called n_epoch times
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

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

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optim.SGD()

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint_" + CLASS + "_" + str(IMAGE_WIDTH) + ".pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        #save_checkpoint(checkpoint, filename="my_checkpoint_" + CLASS + "_" + str(IMAGE_WIDTH) + ".pth.tar")
        save_checkpoint(checkpoint, filename=f"my_checkpoint_{str(LEARNING_RATE)}_{str(NUM_EPOCHS)}_{BATCH_SIZE}_{CLASS}_{str(IMAGE_WIDTH)}.pth.tar")

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        if SAVE_PREDS_AS_IMAGES:
            save_predictions_as_imgs(
                val_loader, model, folder="saved_images/" + CLASS + "/", device=DEVICE
            )


if __name__ == "__main__":
    main()
