from ultralytics import YOLO
import insightface
import cv2
import matplotlib.pyplot as plt


IMAGE_PATH = "input/IMG_0473.jpg"
MODEL_PATH = "models/Face&LandmarksDetector/YOLO26NANO_WARMUP$10&LR0$0.015&LRF$0.004.pt"

image = cv2.imread(IMAGE_PATH)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = YOLO("models/Face&LandmarksDetector/YOLO26NANO_WARMUP$10&LR0$0.015&LRF$0.004.pt")
output = model.predict(IMAGE_PATH)


x, y, w, h = output[0].boxes.xywh[0]
x = int(x-w/2)
y = int(y-h/2)
w = int(w)
h = int(h)



keypoints = output[0].keypoints.xy[0]
kpx1, kpy1 = int(keypoints[0][0]), int(keypoints[0][1])
kpx2, kpy2 = int(keypoints[1][0]), int(keypoints[1][1])
kpx3, kpy3 = int(keypoints[2][0]), int(keypoints[2][1])
kpx4, kpy4 = int(keypoints[3][0]), int(keypoints[3][1])
kpx5, kpy5 = int(keypoints[4][0]), int(keypoints[4][1])


cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 225), 5)
cv2.circle(image, center=(kpx1, kpy1), radius=10, thickness=10, color=(0, 125, 125))
cv2.circle(image, center=(kpx2, kpy2), radius=10, thickness=10, color=(0, 125, 125))
cv2.circle(image, center=(kpx3, kpy3), radius=10, thickness=10, color=(0, 125, 125))
cv2.circle(image, center=(kpx4, kpy4), radius=10, thickness=10, color=(0, 125, 125))
cv2.circle(image, center=(kpx5, kpy5), radius=10, thickness=10, color=(0, 125, 125))



plt.imshow(image)
plt.show()