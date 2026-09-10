import torch
from ultralytics import YOLO
import os
from typing import Literal
from models import get_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from insightface.utils import face_align




class FaceEmbedder:
    def __init__(self, detection_model: Literal["YOLO26NANO_BASE", "YOLO26SMALL_MULTISCALE#0.25&LRF#0.005", "YOLO26NANO_LRF#0.005&COS_LR#TRUE", "YOLO26NANO_WARMUP#10&LR0#0.015&LRF#0.004"], embedding_model: Literal["r18", "r34", "r50", "r100"]):

        BASE_DETECTION_MODELS_DIR = "models/Face&LandmarksDetector"
        BASE_EMBEDDING_MODELS_DIR = "models/FaceEmbedding"

        self.embedding_model_name = embedding_model

        detection_model_ckpt_path = os.path.join(BASE_DETECTION_MODELS_DIR, detection_model, "best.pt")
        embedding_model_weights_path = os.path.join(BASE_EMBEDDING_MODELS_DIR, f"ms1mv3_arcface_{embedding_model}_fp16.pth")

        self.detection_model = self._load_detection_model(detection_model_ckpt_path)
        self.embedding_model = self._load_embedding_model(embedding_model_weights_path)


    def _load_detection_model(self, model_path):
        return YOLO(model_path)

    @torch.no_grad()
    def _embed(self, image):

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).unsqueeze(0).float()
        image.div_(255).sub_(0.5).div_(0.5)
        # net = get_model(name, fp16=False)
        vec = self.embedding_model(image).numpy()
        vec = vec.squeeze()

        return vec

    def _load_embedding_model(self, weights_path):
        weights = torch.load(weights_path, map_location=torch.device("cpu"))
        model = get_model(self.embedding_model_name)
        model.load_state_dict(weights)
        model.eval()
        return model

    def _extract_relevant_info(self, result):
        
        bboxes_xywh = result.boxes.xywh.numpy().round().astype(int)
        bboxes_conf = result.boxes.conf.numpy()
        kps_xy = result.keypoints.xy.numpy().round().astype(int)
        kps_conf = result.keypoints.conf.numpy()

        return bboxes_xywh, bboxes_conf, kps_xy, kps_conf

    def _filter_image_results(self, bboxes_conf, kps_conf):

        bboxes_conf_bin = bboxes_conf>0.5
        kps_conf_bin = (kps_conf>0.5).all(axis=1)
        valid_faces = bboxes_conf_bin & kps_conf_bin
        valid_faces_idxs = np.where(valid_faces)[0]

        return valid_faces_idxs

    def _xy_spatial_sort(self, bboxes):

        bboxes = [[*bboxes[i], i] for i in range(len(bboxes))]
        y_grouping = {point[1]: [] for point in bboxes}

        for (x, y, w, h, i) in bboxes:
            y_grouping[y].append((x, w, h, i))

        for k in y_grouping.keys():
            y_grouping[k].sort(key= lambda p: p[0])


        sorted_points = []

        for k in sorted(list(y_grouping.keys())):
            
            bboxes = [(v[0], k, v[1], v[2], v[3]) for v in y_grouping[k]]
            sorted_points.extend(bboxes)

        sorted_bboxes = [(x, y, w, h) for (x, y, w, h, i) in sorted_points]
        new_indices = [i for (x, y, w, h, i) in sorted_points]
        return sorted_bboxes, new_indices
        



    def embed_faces_from_desk(self, images_paths: list[str]) -> list[np.ndarray]:
        num_images = len(images_paths)
        results = self.detection_model.predict(images_paths)
        outputs = {path: [] for path in images_paths}
        
        
        for idx, result in enumerate(results):
            # extracting only the attribute that matter, the Ultralytics Results object is pure shit
            bboxes_xywh, bboxes_conf, kps_xy, kps_conf= self._extract_relevant_info(result)

             #drop faces with LOW CONFIDENCE in both DETECTIONS/KPS
            valid_faces_indices = self._filter_image_results(bboxes_conf, kps_conf)
            valid_bboxes_xywh = bboxes_xywh[valid_faces_indices].tolist()
            valid_kps_xywh = kps_xy[valid_faces_indices] # wont convert it to a list like the bboxes list yet

            #spatial sorting top-left to bottom right, left to right direction
            sorted_bboxes, new_indices = self._xy_spatial_sort(valid_bboxes_xywh)
            sorted_kps = valid_kps_xywh[new_indices].tolist()


            # for i in range(len(valid_faces_indices)):
            #     image = result.orig_img
            #     x, y, w, h = sorted_bboxes[i]
                
            #     kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5 = np.array(sorted_kps[i]).reshape(10)
            #     cv2.rectangle(image, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 225), 5)
            #     cv2.circle(image, center=(kpx1, kpy1), radius=17, thickness=20, color=(0, 0, 0))
            #     cv2.circle(image, center=(kpx2, kpy2), radius=17, thickness=20, color=(0, 0, 0))
            #     cv2.circle(image, center=(kpx3, kpy3), radius=17, thickness=20, color=(0, 0, 0))
            #     cv2.circle(image, center=(kpx4, kpy4), radius=17, thickness=20, color=(0, 0, 0))
            #     cv2.circle(image, center=(kpx5, kpy5), radius=17, thickness=20, color=(0, 0, 0))
            # plt.imshow(image)
            # plt.show()
            num_faces = valid_faces_indices.shape[0]
            # print(num_faces)
            image = result.orig_img
            # print("IMAGE SHAPE: ", image.shape)
            for i in range(len(valid_faces_indices)):
                x, y, w, h = sorted_bboxes[i]                
                kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5 = np.array(sorted_kps[i]).reshape(10)
                # print("KPX1: ", kpx1)
                kpx1-=x-int(w/2)
                kpx2-=x-int(w/2)
                kpx3-=x-int(w/2)
                kpx4-=x-int(w/2)
                kpx5-=x-int(w/2)
                kpy1-=y-int(h/2)
                kpy2-=y-int(h/2)
                kpy3-=y-int(h/2)
                kpy4-=y-int(h/2)
                kpy5-=y-int(h/2)
                # print("KPX1: ", kpx1)
                face_region = image[int(y-h/2):int(y+h/2), int(x-w/2): int(x+w/2)]
                
                # cv2.rectangle(face_region, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 0, 225), 5)
                # cv2.circle(face_region, center=(kpx1, kpy1), radius=8, thickness=10, color=(0, 0, 0))
                # cv2.circle(face_region, center=(kpx2, kpy2), radius=8, thickness=10, color=(0, 0, 0))
                # cv2.circle(face_region, center=(kpx3, kpy3), radius=8, thickness=10, color=(0, 0, 0))
                # cv2.circle(face_region, center=(kpx4, kpy4), radius=8, thickness=10, color=(0, 0, 0))
                # cv2.circle(face_region, center=(kpx5, kpy5), radius=8, thickness=10, color=(0, 0, 0))
                # plt.imshow(face_region)
                # plt.show()
                aligned = face_align.norm_crop(face_region, landmark=np.array([
                    [kpx1, kpy1],
                    [kpx2, kpy2],
                    [kpx3, kpy3],
                    [kpx4, kpy4],
                    [kpx5, kpy5],
                ]), image_size=112)
                # plt.imshow(aligned)
                # plt.show()
                # face_embedding = embed(self.embedding_model_name)
                # embeddings1 = embed("r18", EMBEDDING_WEIGHTS_PATH, False, face1)
                face_embedding = self._embed(aligned)
                outputs[images_paths[idx]].append((face_embedding, cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)))
            print(f"IMAGE {idx+1}/{len(results)} Done")



        return outputs
                


    def embed_faces(self, images) -> list[np.ndarray]:
        num_images = len(images)
        results = self.detection_model.predict(images)
        outputs = {i: [] for i in range(num_images)}
        
        
        for idx, result in enumerate(results):
            bboxes_xywh, bboxes_conf, kps_xy, kps_conf= self._extract_relevant_info(result)

            valid_faces_indices = self._filter_image_results(bboxes_conf, kps_conf)
            valid_bboxes_xywh = bboxes_xywh[valid_faces_indices].tolist()
            valid_kps_xywh = kps_xy[valid_faces_indices] # wont convert it to a list like the bboxes list yet

            sorted_bboxes, new_indices = self._xy_spatial_sort(valid_bboxes_xywh)
            sorted_kps = valid_kps_xywh[new_indices].tolist()

            num_faces = valid_faces_indices.shape[0]
            image = result.orig_img
            for i in range(len(valid_faces_indices)):
                x, y, w, h = sorted_bboxes[i]                
                kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5 = np.array(sorted_kps[i]).reshape(10)

                kpx1-=x-int(w/2)
                kpx2-=x-int(w/2)
                kpx3-=x-int(w/2)
                kpx4-=x-int(w/2)
                kpx5-=x-int(w/2)
                kpy1-=y-int(h/2)
                kpy2-=y-int(h/2)
                kpy3-=y-int(h/2)
                kpy4-=y-int(h/2)
                kpy5-=y-int(h/2)
                face_region = image[int(y-h/2):int(y+h/2), int(x-w/2): int(x+w/2)]
                
                aligned = face_align.norm_crop(face_region, landmark=np.array([
                    [kpx1, kpy1],
                    [kpx2, kpy2],
                    [kpx3, kpy3],
                    [kpx4, kpy4],
                    [kpx5, kpy5],
                ]), image_size=112)
                face_embedding = self._embed(aligned)
                outputs[idx].append((face_embedding, cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)))
            print(f"IMAGE {idx}/{len(results)} Done")



        return outputs