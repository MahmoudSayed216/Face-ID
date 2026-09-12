# # # import numpy as np
# # # from face_embedder import FaceEmbedder
# # # import cv2

# # # embedder = FaceEmbedder(detection_model="YOLO26NANO_BASE", embedding_model="r34")

# # # IMAGE1 = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/input/17.jpeg"    
# # # IMAGE2 = "/home/mahmoud-sayed/Desktop/Code/Python/Computer Vision/Face Identification/input/190.jpeg"
# # # IMAGE1 = cv2.imread(IMAGE1)
# # # IMAGE2 = cv2.imread(IMAGE2)
# # # # embeddings = embedder.embed_faces_from_desk(images_paths=[
# # # #     IMAGE1,
# # # #     IMAGE2
# # # #     ])

# # # embeddings = embedder.embed_faces(
# # #     images=[IMAGE1, IMAGE2]
# # # )

# # # sayed1 = embeddings[0][0][0]

# # # sayed2 = embeddings[1][1][0]
# # # print(sayed1)
# # # print(sayed2.shape)



# # # cos_sim = np.dot(sayed1, sayed2) / (np.linalg.norm(sayed1) * np.linalg.norm(sayed2))
# # # print("COSINE SIMILARITY = ", cos_sim)

# # class OBJ:
# #     def __init__(self):
# #         self.x = 1


# # def func(obj):
# #     obj.x = 19
# # x = OBJ()

# # print(x.x)
# # func(x)
# # print(x.x)

# import numpy as np


# a = [[1,2], [1,4]]

# ar = np.array(a)

# ar = ar.tolist()

# print(ar)

s = {1,2,4}
z = {1,2,3}

print(z.symmetric_difference(s))