import torch
import torch.nn as nn
from model import tbxrayCNN
from data import get_dataloaders, get_transforms, all_images

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tbxrayCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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