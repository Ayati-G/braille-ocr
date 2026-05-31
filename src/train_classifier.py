import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_SAVE_PATH = Path(__file__).parent.parent / "models" / "mobilenet"

NUM_CLASSES = 26      # how many letters are we classifying?
BATCH_SIZE = 32       # images processed at once — bigger = faster but more VRAM
NUM_EPOCHS = 15      # how many times to loop through the full dataset
LEARNING_RATE = 0.001  # how fast the model updates weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

train_transforms = transforms.Compose([
    transforms.Resize((224,224)), # MobileNetV3 compatible
    transforms.RandomRotation(8), #randomly tilting upto 8 deg
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #20% var in brightness and contrast
    transforms.ToTensor(), #convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # standard ImageNet values, don't change
                         std=[0.229, 0.224, 0.225])

])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dir = DATA_DIR / "train"
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms) #ImgFolder automatically maps each folder to label 'a': 0
val_dir = DATA_DIR / "val"
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #shuffle so that all alphabets are shuffled
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) #num of workers to bring img to gpu so it aint staying idle

#model dir
MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
# -------------- Class mapping -----------#
import json
class_to_idx = train_dataset.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}
with open(MODEL_SAVE_PATH / "class_mapping.json", "w") as f:
    json.dump(idx_to_class, f)

# model = models.mobilenet_v3_small(weights=none)
# #model.classifier[3] = nn.Linear(?, NUM_CLASSES)
# print(model.classifier)
state_dict = torch.load(r"C:\Users\Ayati Gupta\.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth")
model = models.mobilenet_v3_small(weights=None)
model.load_state_dict(state_dict)

# print(model.classifier)
# 0-2: identify features
# layer 3: classifier {the one we are updating}
model.classifier[3] = nn.Linear(1024,NUM_CLASSES) #nn.Linear(input,out)
model = model.to(device) # moving model to gpu/cpu

criterion = nn.CrossEntropyLoss() #CrossEntropyLoss for multi-classes
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) #adam is a fast optimizer that adjusts weights to reduce loss

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for images,labels in train_loader:
        #move batch_size imgs and labels to device
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() #erase prev gradient accumulated by Pytorch
        outputs = model(images) #[a_score...z_score]*32 --> 32x26
        loss = criterion(outputs, labels) #compare prediction to labels
        loss.backward() #backward propagation
        optimizer.step() #update weights

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}- Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            # move to device
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(images)
            # get predicted class (hint: torch.argmax(outputs, dim=1))
            predictions = torch.argmax(outputs, dim=1)
            # compare to labels, count correct
            correct += (predictions ==labels).sum()
            total += labels.size(0)


    print(f"Val Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), MODEL_SAVE_PATH / "mobilenet_braille.pth")
print("Model saved!")


