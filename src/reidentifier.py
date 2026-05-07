import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2

class PlayerReidentifier:
    """
    Extracts visual embeddings from player crops
    using a pretrained ResNet50 as a backbone.
    """
    def __init__(self, device="cpu"):
        self.device = device

        # Pretrained ResNet50 — we remove the last classification layer
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.eval()
        self.model.to(device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),   # standard re-id size
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Gallery: {tracker_id: [embedding1, embedding2, ...]}
        self.gallery = {}

    def extract_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Converts a BGR crop into a 2048-dimensional vector."""
        if crop_bgr.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor).squeeze().cpu().numpy()
        return emb / np.linalg.norm(emb)   # normalize to unit length

    def update_gallery(self, tracker_id: int, crop_bgr: np.ndarray):
        """Saves the embedding of a known player."""
        emb = self.extract_embedding(crop_bgr)
        if emb is None:
            return
        if tracker_id not in self.gallery:
            self.gallery[tracker_id] = []
        # Store maximum 10 embeddings per player
        if len(self.gallery[tracker_id]) < 10:
            self.gallery[tracker_id].append(emb)

    def search_in_gallery(self, crop_bgr: np.ndarray, threshold: float = 0.7):
        """
        Compares a crop with the gallery.
        Returns the most similar tracker_id if it exceeds the threshold,
        or None if there is no match.
        """
        if not self.gallery:
            return None

        emb_query = self.extract_embedding(crop_bgr)
        if emb_query is None:
            return None

        best_id = None
        best_sim = -1

        for tid, embeddings in self.gallery.items():
            # Average similarity with all stored embeddings of this player
            sims = [np.dot(emb_query, e) for e in embeddings]
            mean_sim = np.mean(sims)
            if mean_sim > best_sim:
                best_sim = mean_sim
                best_id = tid

        if best_sim >= threshold:
            return best_id
        return None   # no sufficiently good match found
