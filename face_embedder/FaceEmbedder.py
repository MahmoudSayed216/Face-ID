import torch
from ultralytics import YOLO
import os
import time
from typing import Literal
from models import get_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from insightface.utils import face_align



## TODO: BATCH EMBED FACES PER IMAEG
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
        s = time.time()
        vec = self.embedding_model(image).numpy()
        print("PURE EMBEDDING TIME = ", time.time() - s)
        vec = vec.squeeze()

        return vec
# PURE EMBEDDING                        TIME =  0.03637528419494629
# FaceEmbedder.embed_faces.embed        TIME =  0.03657793998718262
    def _load_embedding_model(self, weights_path):
        weights = torch.load(weights_path, map_location=torch.device("cpu"))
        model = get_model(self.embedding_model_name, fp16=True)
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
                x = int(x - w/2)
                y = int(y - h/2)
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
                face_region = image[int(y):int(y+h), int(x): int(x+w)]
                keypoints = [
                    [kpx1, kpy1],
                    [kpx2, kpy2],
                    [kpx3, kpy3],
                    [kpx4, kpy4],
                    [kpx5, kpy5],
                ]
                aligned = face_align.norm_crop(face_region, landmark=np.array(keypoints), image_size=112)
                # plt.imshow(aligned)
                # plt.show()
                # face_embedding = embed(self.embedding_model_name)
                # embeddings1 = embed("r18", EMBEDDING_WEIGHTS_PATH, False, face1)
                face_embedding = self._embed(aligned)
                outputs[images_paths[idx]].append({"embeddings":face_embedding, "face":cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB), "bbox":(x,y,w,h), "kps": keypoints})

            print(f"IMAGE {idx+1}/{len(results)} Done")



        return outputs
                


    def embed_faces(self, images) -> list[np.ndarray]:
        start = time.time()
        num_images = len(images)
        time_det_s = time.time()
        results = self.detection_model.predict(images)
        time_det_e = time.time()
        outputs = {i: [] for i in range(num_images)}
        
        
        for idx, result in enumerate(results):

            time_eri_s = time.time()
            bboxes_xywh, bboxes_conf, kps_xy, kps_conf= self._extract_relevant_info(result)
            time_eri_e = time.time()

            time_fir_s = time.time()
            valid_faces_indices = self._filter_image_results(bboxes_conf, kps_conf)
            time_fir_e = time.time()
            valid_bboxes_xywh = bboxes_xywh[valid_faces_indices].tolist()
            valid_kps_xywh = kps_xy[valid_faces_indices] # wont convert it to a list like the bboxes list yet
            time_xyss_s = time.time()
            sorted_bboxes, new_indices = self._xy_spatial_sort(valid_bboxes_xywh)
            time_xyss_e = time.time()
            sorted_kps = valid_kps_xywh[new_indices].tolist()

            num_faces = valid_faces_indices.shape[0]
            image = result.orig_img
            time_loop_s = time.time()
            time_align_total = 0
            time_embed_total = 0
            for i in range(len(valid_faces_indices)):
                
                x, y, w, h = sorted_bboxes[i]
                kpx1, kpy1, kpx2, kpy2, kpx3, kpy3, kpx4, kpy4, kpx5, kpy5 = np.array(sorted_kps[i]).reshape(10)

                x = int(x - w/2)
                y = int(y - h/2)
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
                
                face_region = image[int(y):int(y+h), int(x): int(x+w)]
                keypoints = [
                    [kpx1, kpy1],
                    [kpx2, kpy2],
                    [kpx3, kpy3],
                    [kpx4, kpy4],
                    [kpx5, kpy5],
                ]
                time_align_s = time.time()
                aligned = face_align.norm_crop(face_region, landmark=np.array(keypoints), image_size=112)
                
                time_align_e = time.time()
                time_embedding_s = time.time()
                face_embedding = self._embed(aligned)
                time_embedding_e = time.time()
                print("TYPE OF EMBEDDING DS: ", type(face_embedding))
                print("SHAPE OF EMBEDDING DS: ", face_embedding.shape)
                
                # face_region.resize(new_indices(128, 128, 3))
                
                outputs[idx].append({"embeddings":face_embedding, "face":cv2.cvtColor(cv2.resize(face_region, (128, 128)), cv2.COLOR_BGR2RGB), "bbox":(x,y,w,h), "kps": keypoints})
                time_align_total += (time_align_e - time_align_s)
                time_embed_total += (time_embedding_e - time_embedding_s)

        time_loop_e = time.time()

        time_eri_total         = round(time_eri_e - time_eri_s, 4)
        time_fir_total         = round(time_fir_e - time_eri_s, 4)
        time_xyss_total        = round(time_xyss_e - time_xyss_s, 4)
        time_loop_total        = round(time_loop_e - time_loop_s, 4)
        time_det_total         = round(time_det_e - time_det_s, 4)
        time_embed_faces_total = round(time.time() - start, 4)

        print("FaceEmbedder.embed_faces.detect       TIME = ", time_det_total*1000, "ms")
        print("FaceEmbedder.embed_faces.embed        TIME = ", time_embed_total*1000, "ms")
        print("FaceEmbedder._extract_relevant_info() TIME = ", time_eri_total*1000, "ms")
        print("FaceEmbedder._filter_image_results()  TIME = ", time_fir_total*1000, "ms")
        print("FaceEmbedder._xy_spatial_sort()       TIME = ", time_xyss_total*1000, "ms")
        print("FaceEmbedder.embed_faces.loop         TIME = ", time_loop_total*1000, "ms")
        print("FaceEmbedder.embed_faces.loop.align   TIME = ", time_align_total*1000, "ms")
        print("TOTAL_________________________________TIME = ", time_embed_faces_total*1000, "ms")
        print("TOTAL MINUS INTERNAL FUNCTION CALLS   TIME = ", (time_embed_faces_total - (time_eri_total+time_fir_total+time_xyss_total))*1000, "ms")


        return outputs