import torch
from torch.backends import cudnn

from qat import EfficientDetBackbone
import cv2
import matplotlib.pyplot as plt
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess

# Configuration
compound_coef = 0
force_input_size = None  # Set None to use the default size
img_path = "D:/QAT/EfficientDet/QAT_ED_PyTorch/test/img.png"
weight_file = "D:\QAT\EfficientDet\QAT_ED_PyTorch\weights\Efficientdet-d0-qat.pth"

threshold = 0.2
iou_threshold = 0.2

# use_cuda = torch.cuda.is_available()
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

# Preprocessing
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

try:
    ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)
except FileNotFoundError:
    raise ValueError(f"Image not found at {img_path}")

# Convert to tensor
x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0).to(device)
x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

# Model setup
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                             ratios=[(1.0, 1.0), (1.3, 0.8), (1.9, 0.5)],
                             scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

torch.ao.quantization.prepare_qat(model, inplace=True)

quantized_model = torch.ao.quantization.convert(model.eval(), inplace=True)
quantized_model.eval()

try:
    quantized_model.load_state_dict(torch.load(weight_file, map_location=device), strict=False)
except FileNotFoundError:
    raise ValueError(f"Model weights not found at {weight_file}")

quantized_model.requires_grad_(False).eval().to(device)

if use_float16:
    quantized_model = quantized_model.half()

# Inference
with torch.no_grad():
    features, regression, classification, anchors = quantized_model(x)

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    out = postprocess(x,
                      anchors, regression, classification,
                      regressBoxes, clipBoxes,
                      threshold, iou_threshold)

out = invert_affine(framed_metas, out)

# Visualization
for i in range(len(ori_imgs)):
    if len(out[i]['rois']) == 0:
        print(f"No objects detected in image {i}")
        continue
    ori_imgs[i] = ori_imgs[i].copy()
    for j in range(len(out[i]['rois'])):
        (x1, y1, x2, y2) = out[i]['rois'][j].astype(int)
        cv2.rectangle(ori_imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
        obj = obj_list[out[i]['class_ids'][j]]
        score = float(out[i]['scores'][j])
        cv2.putText(ori_imgs[i], f'{obj}, {score:.3f}', (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Convert BGR to RGB for displaying with matplotlib
    plt.imshow(cv2.cvtColor(ori_imgs[i], cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()
