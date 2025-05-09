import torch
from torchvision import models, transforms
from PIL import Image
import os

MODEL_PATH = 'best_model.pth'
INPUT_FOLDER = 'cmqs'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(INPUT_FOLDER, filename)

        try:
            img = Image.open(image_path).convert('RGB')
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()

            if pred == 1:
                img = img.rotate(180)
                img.save(image_path)

        except:
            pass
