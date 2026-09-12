from face_embedder import FaceEmbedder
from dotenv import load_dotenv
import os
import time
import cv2
from typing import Literal
from FaceIDSystem.InterfaceWrapper import CameraViewerWrapper
from FaceIDSystem.QdrantClientWrapper import QdrantClientWrapper
from PIL import Image



class FaceIDSystem:
    def __init__(
                self,  
                fps = "auto",
                detection_model: Literal["YOLO26NANO_BASE", "YOLO26SMALL_MULTISCALE#0.25&LRF#0.005", "YOLO26NANO_LRF#0.005&COS_LR#TRUE", "YOLO26NANO_WARMUP#10&LR0#0.015&LRF#0.004"] = "YOLO26NANO_BASE", 
                embedding_model: Literal["r18", "r34", "r50", "r100"] = "r18"
                ):
        load_dotenv()
        self.embedder = FaceEmbedder(detection_model=detection_model, embedding_model=embedding_model)
        self.qdrant_client = QdrantClientWrapper(db_path=os.getenv("DB_PATH"), collection_name=os.getenv("COLLECTION_NAME"))
        self.interface = CameraViewerWrapper(function= self._operate_on_frame)
        self.past_frame_identities = {}

    def _operate_on_frame(self, frame):
        # SHOULD BE THREADED
        start = time.time()
        embeddings = self.embedder.embed_faces([frame])
        frame_results = embeddings[0] # has N faces
        lst = []
        for result in frame_results:
            x, y, w, h = result["bbox"]
            embeddings = result["embeddings"]
            face = result["face"]
            kps = result["kps"]
            lst.append((embeddings, face))
            # self._annotate_face() #UTILS
            self.annotate_face(frame, x, y, w, h, kps)

        time_db_search_s = time.time()
        # hits = []
        current_frame_identities = {}
        for emb, face in lst:
            _id, score, path, name = self.qdrant_client.search(emb)
            # if _id not in hits:
            # current_frame_identities.add((_id, score, path, name))
            current_frame_identities[_id] = {"score": score, "path": path, "name": name}
        time_db_search_e = time.time()
        time_db_search_total = (time_db_search_e - time_db_search_s) * 1000
        print("cfi keys: ", list(current_frame_identities.keys()))
        print("FaceIDSystem._operate_on_fram.dbsearch   TIME = ", time_db_search_total, "ms")
        print("prio: ", list(self.past_frame_identities.keys()))
        to_read = current_frame_identities.keys() - self.past_frame_identities.keys()
        to_remove = self.past_frame_identities.keys() - current_frame_identities.keys()
        print("TO READ: ", to_read)
        print("TO REMOVE: ", to_remove)
        for key in to_remove:
            self.past_frame_identities.pop(key)
        i = 0
        for k in to_read:
            print(f"KEY{i}: ", k)
            i+=1
            _id
            self.past_frame_identities[k] = current_frame_identities[k]
            picture_path = self.past_frame_identities[k]["path"]
            image = Image.open(picture_path)
            image = image.resize(size=(150, 150))
            self.past_frame_identities[k]["face"] = image
            print("post [within loop]: ",list(self.past_frame_identities.keys()))
        print("post: ",list(self.past_frame_identities.keys()))
        return frame, self.past_frame_identities

    def annotate_face(self, frame, x, y, w, h, kps):
        cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=10)
        for i in range(5):
            kpx, kpy = kps[i]
            kpx = kpx + x - int(w/2)
            kpy = kpy + y - int(h/2)
            cv2.circle(frame, center=(kpx, kpy), radius=10, thickness=5, color=(0, 255, 0))
        

    def run(self):
        self.interface.run()
    

