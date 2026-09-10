from FaceEmbedder import FaceEmbedder
import numpy as np



embedder = FaceEmbedder(detection_model="YOLO26NANO_BASE", embedding_model="r34")

IMAGE_1 = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/input/17.jpeg"    
IMAGE_2 = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/input/9.jpeg"

embeddings = embedder.embed_faces_from_desk(images_paths=[
    IMAGE_1,
    IMAGE_2
    ])


print(embeddings)

sayed1 = embeddings[IMAGE_1][1]
sayed2 = embeddings[IMAGE_2][0]




cos_sim = np.dot(sayed1, sayed2) / (np.linalg.norm(sayed1) * np.linalg.norm(sayed2))
print("COSINE SIMILARITY = ", cos_sim)

