import torch
import timm
import cv2
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import urllib.request
import os

# Step 1: Load the model
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
model.eval()

# Step 2: Create preprocessing pipeline
config = resolve_data_config({}, model=model)
transform = create_transform(**config)

# Step 3: Download ImageNet labels (once)
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
LABELS_FILE = "imagenet_classes.txt"
if not os.path.exists(LABELS_FILE):
    urllib.request.urlretrieve(LABELS_URL, LABELS_FILE)

with open(LABELS_FILE, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Step 4: Capture image from webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not access the webcam.")
    exit()

print("📸 Press SPACE to take a photo...")
while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam - Press SPACE to capture", frame)
    key = cv2.waitKey(1)
    if key % 256 == 32:  # SPACE key
        img = frame
        break

cap.release()
cv2.destroyAllWindows()

# Step 5: Convert to PIL, preprocess, and predict
image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')
tensor = transform(image).unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    out = model(tensor)
    probabilities = torch.nn.functional.softmax(out[0], dim=0)

# Step 6: Get top-5 predictions
top5_prob, top5_catid = torch.topk(probabilities, 5)

print("\n🎯 Top-5 Predictions:")
for i in range(top5_prob.size(0)):
    label = categories[top5_catid[i]]
    score = top5_prob[i].item()
    print(f"{label}: {score:.4f}")
