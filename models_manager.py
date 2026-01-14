import onnxruntime as ort
import os

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.text_session = None
        self.vision_session = None
        self.lama_session = None
        self.removebg_session = None
        # Priority: CUDA -> CPU
        self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
    def load_clip_models(self):
        if self.text_session and self.vision_session:
            return

        # Paths expected in model/ directory
        text_model_path = os.path.join("model", "vit-b-16.txt.fp32.onnx")
        vision_model_path = os.path.join("model", "vit-b-16.img.fp32.onnx")
        
        if not os.path.exists(text_model_path) or not os.path.exists(vision_model_path):
             print(f"Warning: CLIP models not found in model/ directory. Expected: {text_model_path}, {vision_model_path}")
             return

        print(f"Loading CLIP models from {text_model_path} and {vision_model_path}...")
        try:
            self.text_session = ort.InferenceSession(text_model_path, providers=self.providers)
            self.vision_session = ort.InferenceSession(vision_model_path, providers=self.providers)
            print("CLIP models loaded successfully.")
        except Exception as e:
            print(f"Error loading CLIP models: {e}")
            raise e

    def get_clip_sessions(self):
        self.load_clip_models()
        return self.text_session, self.vision_session

    def load_lama_model(self):
        # Placeholder for LaMa model loading
        pass

    def load_removebg_model(self):
        # Placeholder for RMBG model loading
        pass

model_manager = ModelManager()
