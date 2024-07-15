import torch
import torchvision.models as models
import os
from PIL import Image
from torchvision import transforms

# Load the pre-trained ResNet-34 model
resnet34 = models.resnet34(pretrained=True)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img = Image.open("/home/rchen2/CAT/Brown_Field/Train/imgs/img_1.png")
img_t = preprocess(img)
batch_t = torch.unsqueeze(img_t, 0)

# Put the model in evaluation mode
resnet34.eval()

# Perform inference
with torch.no_grad():
    out = resnet34(batch_t)

# Print the output
print(out)
