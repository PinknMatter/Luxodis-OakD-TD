import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Normalize, ToTensor

class DepthSegmenter:
    def __init__(self, model_type='DPT_Hybrid'):
        self.model = torch.hub.load('intel-isl/MiDaS', model_type)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], 
                     std=[0.229, 0.224, 0.225])
        ])

    def process_frame(self, frame):
        # Preprocessing
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (384, 384))
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode='bicubic',
                align_corners=False
            ).squeeze()
        
        # Postprocessing
        depth_map = prediction.cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        depth_map = (depth_map * 255).astype(np.uint8)
        
        return depth_map

# TouchDesigner Operator Implementation
class DepthOperator:
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.depth_processor = DepthSegmenter()
        
        # Add parameters
        self.threshold = ownerComp.appendParFloat('Threshold', label='Depth Threshold', default=0.5)
        self.color_map = ownerComp.appendParMenu('Colormap', label='Color Map', names=['Viridis', 'Plasma', 'Inferno', 'Magma'])
        
    def ProcessFrame(self, frame):
        depth_map = self.depth_processor.process_frame(frame)
        
        # Apply threshold
        thresholded = (depth_map > (self.threshold * 255)).astype(np.uint8) * 255
        
        # Apply color map
        cmap = {
            0: cv2.COLORMAP_VIRIDIS,
            1: cv2.COLORMAP_PLASMA,
            2: cv2.COLORMAP_INFERNO,
            3: cv2.COLORMAP_MAGMA
        }[self.color_map]
        
        colored = cv2.applyColorMap(thresholded, cmap)
        
        return colored

# Required TouchDesigner callbacks
def onInitialize(ownerComp):
    return DepthOperator(ownerComp)

def onStart(ownerComp):
    pass

def whileRunning(ownerComp):
    pass

def onDone(ownerComp):
    pass
