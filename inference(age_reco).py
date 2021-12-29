#inference
import cv2
import numpy as np

model.eval()

img = cv2.imread("images1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img,(70,70))
tensor = transform(img).unsqueeze(0).to(device)

preds = model(tensor)

preds = preds.cpu().detach().numpy()

print(preds)