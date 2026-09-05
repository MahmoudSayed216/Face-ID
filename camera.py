import cv2
from ultralytics import YOLO
from insightface.utils import face_align
import numpy as np



cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FRAME WIDTH: ", frame_width)
print("FRAME HEIGHT: ", frame_height)
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
MODEL_PATH = "models/Face&LandmarksDetector/YOLO26NANO_WARMUP$10&LR0$0.015&LRF$0.004.pt"
model = YOLO("models/Face&LandmarksDetector/YOLO26NANO_WARMUP$10&LR0$0.015&LRF$0.004.pt")

def get_bbox_and_points(output):
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

    return x, y, w, h, kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5


while True:
    ret, frame = cam.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = model.predict(image)
    x, y, w, h, kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5 = get_bbox_and_points(output)
    landmarks = np.array([
        [kpx1, kpy1],
        [kpx2, kpy2],
        [kpx3, kpy3],
        [kpx4, kpy4],
        [kpx5, kpy5],
    ])



    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 225), 5)
    cv2.circle(image, center=(kpx1, kpy1), radius=5, thickness=5, color=(0, 125, 125))
    cv2.circle(image, center=(kpx2, kpy2), radius=5, thickness=5, color=(0, 125, 125))
    cv2.circle(image, center=(kpx3, kpy3), radius=5, thickness=5, color=(0, 125, 125))
    cv2.circle(image, center=(kpx4, kpy4), radius=5, thickness=5, color=(0, 125, 125))
    cv2.circle(image, center=(kpx5, kpy5), radius=5, thickness=5, color=(0, 125, 125))
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    aligned = face_align.norm_crop(image, landmark=landmarks, image_size=112)
    # aligned = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

    image[:112, :112] = aligned

    cv2.imshow('Camera', image)
    
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()