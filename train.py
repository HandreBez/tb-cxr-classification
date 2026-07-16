import torch
import torch.nn as nn
from model import tbxrayCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tbxrayCNN()
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(15):
    for images, labels in train_loader:

        images  = images.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()