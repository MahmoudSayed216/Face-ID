from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

class QdrantClientWrapper:
    def __init__(self, db_path, collection_name):
        print("INITIALIZING QDRANT")
        self.qdrant_client = QdrantClient(path=db_path)
        self.incremental_id = 22
        self.collection_name = collection_name
        if not self.qdrant_client.collection_exists(collection_name=collection_name):
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )

    def insert(self, embedding: list[float], face_path: str = "",  name=""):
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=self.incremental_id,
                vector=embedding,
                payload={
                    "face_path": face_path,
                    "name": name
                    }
            )]
        )
        self.incremental_id+=1

    def search(self, embedding):
        hits = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=embedding,
            limit=2
        ).points
        print("LEN(HITS)", len(hits))
        mx = -1
        i_mx = -1
        for i, h in enumerate(hits):
            print("ID: ", h.id)
            print("SCORE: ", h.score)
            if h.score > mx:
                mx = h.score
                i_mx = i
        print("RETURNED: ", hits[i_mx].id)
        return hits[i_mx].id, hits[i_mx].score, hits[i_mx].payload["face_path"], hits[i_mx].payload["name"]

    