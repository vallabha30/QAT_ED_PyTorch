import torch
from torch.backends import cudnn
import cv2
import numpy as np
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess
from qat import EfficientDetBackbone

# Configuration
compound_coef = 0
threshold = 0.2
iou_threshold = 0.2
use_cuda = False
use_float16 = False
cudnn.fastest = use_cuda
cudnn.benchmark = use_cuda

device = torch.device('cuda' if use_cuda else 'cpu')

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Model setup
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

torch.ao.quantization.prepare_qat(model, inplace=True)
quantized_model = torch.ao.quantization.convert(model.eval(), inplace=True)
quantized_model.eval()
weight_file = "D:/QAT/EfficientDet/QAT_ED_PyTorch/weights/Efficientdet-d0-qat.pth"

try:
    quantized_model.load_state_dict(torch.load(weight_file, map_location=device), strict=False)
except FileNotFoundError:
    raise ValueError(f"Model weights not found at {weight_file}")

quantized_model.requires_grad_(False).eval().to(device)

if use_float16:
    quantized_model = quantized_model.half()


# Preprocessing for live frames
def preprocess_frame(frame, max_size=512):
    h, w, _ = frame.shape
    scale = max_size / max(h, w)
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    padded_frame = np.zeros((max_size, max_size, 3), dtype=np.uint8)
    padded_frame[:resized_frame.shape[0], :resized_frame.shape[1], :] = resized_frame
    meta = {
        'new_width': resized_frame.shape[1],
        'new_height': resized_frame.shape[0],
        'original_width': w,
        'original_height': h,
        'padding_width': max_size - resized_frame.shape[1],
        'padding_height': max_size - resized_frame.shape[0],
        'scale': scale
    }
    return padded_frame, meta

# Video capture setup
cap = cv2.VideoCapture(0)  # Change to video file path if needed

if not cap.isOpened():
    raise RuntimeError("Error: Unable to access the camera!")

# Real-time detection
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Preprocess the frame
        framed_img, framed_meta = preprocess_frame(frame, max_size=512)
        x = torch.from_numpy(framed_img).to(device).to(torch.float32 if not use_float16 else torch.float16)
        x = x.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            features, regression, classification, anchors = quantized_model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x, anchors, regression, classification,
                              regressBoxes, clipBoxes, threshold, iou_threshold)
            out = invert_affine([framed_meta], out)

        # Draw detections on the frame
        for i in range(len(out[0]['rois'])):
            (x1, y1, x2, y2) = out[0]['rois'][i].astype(int)
            obj = obj_list[out[0]['class_ids'][i]]
            score = float(out[0]['scores'][i])
            scale = framed_meta['scale']
            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f'{obj}, {score:.3f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Display the frame
        cv2.imshow('EfficientDet Live Object Detection', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
