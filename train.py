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
        '''
        Calling .backward() mutiple times accumulates the gradient (by addition) for each parameter. This is why you should call optimizer.zero_grad() after each .step() call.
        '''
        scaler.step(optimizer)
        scaler.update()
        '''
        -> optimizer.step() performs a parameter update based on the current gradient (stored in .grad attribute of a parameter) 
        
        -> When you call loss.backward(), all it does is compute gradient of loss w.r.t all the parameters in loss that have requires_grad = True and store them in parameter.grad attribute for every parameter.
        optimizer.step() updates all the parameters based on parameter.grad
        
        -> The optimizer takes the parameters we want to update, the learning rate we want to use (and possibly many other parameters as well, and performs the updates through its step() method
        '''
        '''
        =============================================================================================
        Recall that when initializing optimizer you explicitly tell it what parameters (tensors) of the model it should be updating. The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you call backward() on the loss. After computing the gradients for all tensors in the model, calling optimizer.step() makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored grad to update their values.
        '''
        '''
        important nouns:
        gradient: 
        An error gradient is the direction and magnitude calculated during the training of a neural network that is used to update the network weights in the right direction and by the right amount. 
        To calculate the gradient of a straight line we choose two points on the line itself. The difference in height (y co-ordinates) ÷ The difference in width (x co-ordinates). If the answer is a positive value then the line is uphill in direction. If the answer is a negative value then the line is downhill in direction.
        optimizer (Adam, SGD, RMSProp, ...):
        An optimizer is a method or algorithm to update the various parameters that can reduce the loss in much less effort.
        '''
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