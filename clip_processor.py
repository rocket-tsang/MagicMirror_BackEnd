import numpy as np
from PIL import Image
import io
import cn_clip.clip as clip
from models_manager import model_manager
import torchvision.transforms as transforms

class ClipProcessor:
    def __init__(self):
        # Preprocessing transform
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711))
        ])

    def _preprocess_image(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0)
            return image_tensor.numpy()
        except Exception as e:
            raise ValueError(f"Invalid image file: {str(e)}")

    def get_text_features(self, text_list):
        text_session, _ = model_manager.get_clip_sessions()
        if not text_session:
            raise RuntimeError("CLIP Text model not loaded")
            
        features = []
        for text in text_list:
            # Tokenize (context_length default 52)
            token = clip.tokenize([text], context_length=52).numpy().astype(np.int64)
            # Inference
            feature = text_session.run(None, {'text': token})[0]
            # L2 Normalize
            feature = feature / np.linalg.norm(feature)
            features.append(feature)
        return np.vstack(features)

    def get_image_features(self, image_bytes):
        _, vision_session = model_manager.get_clip_sessions()
        if not vision_session:
             raise RuntimeError("CLIP Vision model not loaded")

        image_data = self._preprocess_image(image_bytes)
        # Inference
        image_features = vision_session.run(None, {'image': image_data.astype(np.float32)})[0]
        # L2 Normalize
        image_features = image_features / np.linalg.norm(image_features)
        return image_features

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, image_bytes, texts):
        """
        Generic prediction: Image vs List of Texts
        Returns probabilities for each text.
        """
        img_feat = self.get_image_features(image_bytes)
        txt_feats = self.get_text_features(texts)
        
        # Dot product
        logits = np.dot(img_feat, txt_feats.T)[0]
        # Softmax with logit scale (100 is standard for CLIP)
        probs = self.softmax(logits * 100)
        
        return probs.tolist(), logits.tolist()

clip_processor = ClipProcessor()
