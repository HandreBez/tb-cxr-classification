import io
import torch
from PIL import Image
from model import ResNet18TBPretrained
from data import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet18TBPretrained()
model.load_state_dict(torch.load("resnet18_pretrained_weights.pt", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

_, transform = get_transforms(augment=False)

def predict(image_bytes: bytes) -> float:
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        tb_prob = probs[:, 1]

    return tb_prob.item()

if __name__ == "__main__":
    with open("data/cache/Shenzhen/CHNCXR_0001_0.png", "rb") as f:
        image_bytes = f.read()
    print(predict(image_bytes))