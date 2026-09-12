from FaceIDSystem.QdrantClientWrapper import QdrantClientWrapper
from face_embedder import FaceEmbedder
from PIL import Image
import numpy as np


embedder = FaceEmbedder(detection_model='YOLO26NANO_BASE', embedding_model='r18') 
client = QdrantClientWrapper(db_path="./Database", collection_name="Faces")


IMAGE = "./input/22.jpeg"


embeddings = embedder.embed_faces_from_desk(images_paths=[IMAGE])

first_image_embeddings = embeddings[IMAGE]

first_face_embedding = first_image_embeddings[0]["embeddings"].tolist()
print(first_image_embeddings)
first_face_picture = first_image_embeddings[0]["face"].tolist()

# client.insert(embedding=first_face_embedding, face=first_face_picture, name="Sayed")
image = np.asarray(first_face_picture, dtype=np.uint8)
image = Image.fromarray(image)
image.save("./identities/22.jpeg")

client.insert(embedding=first_face_embedding, name="Amr", face_path="./identities/22.jpeg")