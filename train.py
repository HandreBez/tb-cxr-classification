import torch
import torch.nn as nn
from model import tbxrayCNN
from model import ResNet18TB
from model import ResNet18TBPretrained
from model import DenseNet121TB
from model import DenseNet121TBScratch
from data import get_dataloaders, get_transforms, all_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = tbxrayCNN()
#model = ResNet18TB()
model = ResNet18TBPretrained()
#model = DenseNet121TB()
#model = DenseNet121TBScratch()
model = model.to(device)


run_name = "resnet18_pretrained"

criterion = nn.CrossEntropyLoss()

use_differential_lr = True   # flip to False to use a single uniform LR for the active model

if use_differential_lr:
    backbone_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if name.startswith("resnet.fc"):
            fc_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 0.0001},
        {"params": fc_params, "lr": 0.001},
    ])
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def evaluate(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct, total = 0, 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            predictions = logits.argmax(dim=1)

            probs = torch.softmax(logits, dim=1)
            tb_probs = probs[:, 1]

            all_probs.extend(tb_probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())


            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total * 100

    return avg_loss, accuracy, all_labels, all_probs

if __name__ == "__main__":
    train_loader, test_loader = get_dataloaders(all_images, num_workers=6, augment=True)

    for epoch in range(15):
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:

            images  = images.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            total += labels.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            

        avg_loss = running_loss / len(train_loader)

        print(f"Epoch: {epoch}  Avg Loss: {avg_loss} Accuracy: %{correct/total*100} ")
    
    test_loss, test_accuracy, all_labels, all_probs = evaluate(model, test_loader, criterion, device)
    torch.save({"labels": all_labels, "probs": all_probs}, f"{run_name}_predictions.pt")
    torch.save(model.state_dict(), f"{run_name}_weights.pt")
    print(f"\nTest Loss: {test_loss:.4f}  Test Accuracy: {test_accuracy:.2f}%")


