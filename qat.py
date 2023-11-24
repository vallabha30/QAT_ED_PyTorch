import os
import sys
import time
import numpy as np
from torch.serialization import MAP_LOCATION
from tqdm.autonotebook import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import yaml
import json

from torch import nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch
from torchvision.ops.boxes import nms as nms_torch
from torch.ao.quantization import QuantStub, DeQuantStub
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientnet import EfficientNet 
from efficientnet.utils import MemoryEfficientSwish, Swish
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from efficientdet.model import BiFPN, Regressor, QAT_Classifier, EfficientNet
from efficientdet.utils import Anchors
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding

from torchvision.datasets import CocoDetection
from torch.nn.functional import interpolate

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)


#<---------MODEL ARCHITECTURE----------->



class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
        self.quant=QuantStub()
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
        self.dequant=DeQuantStub()
        

    def forward(self, x):
        x = self.quant(x)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)
        x=self.dequant(x)
        return x

class EfficientDetBackbone(nn.Module):
    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.quant=QuantStub()
        self.compound_coef = compound_coef

        self.backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
        self.pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
        self.anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5., 4.]
        self.aspect_ratios = kwargs.get('ratios', [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])
        self.num_scales = len(kwargs.get('scales', [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[BiFPN(self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7)
              for _ in range(self.fpn_cell_repeats[compound_coef])])

        self.num_classes = num_classes
        self.regressor = Regressor(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                   num_layers=self.box_class_repeats[self.compound_coef],
                                   pyramid_levels=self.pyramid_levels[self.compound_coef])
        self.classifier = QAT_Classifier(in_channels=self.fpn_num_filters[self.compound_coef], num_anchors=num_anchors,
                                     num_classes=num_classes,
                                     num_layers=self.box_class_repeats[self.compound_coef],
                                     pyramid_levels=self.pyramid_levels[self.compound_coef])

        self.anchors = Anchors(anchor_scale=self.anchor_scale[compound_coef],
                               pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                               **kwargs)

        self.backbone_net = EfficientNet(self.backbone_compound_coef[compound_coef], load_weights)
        self.dequant=DeQuantStub()
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        

        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        #features=self.quant(features)

        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)
        #features=self.dequant(features)

        return features , regression, classification, anchors
      
        
    def fuse_model(self):
        fuse_modules = torch.quantization.fuse_modules

        # Fuse BiFPN layers
        #for m in self.bifpn.modules():
            #if type(m) == BiFPN:
                #fuse_modules(m, ['conv', 'bn'], inplace=True)

        # Fuse Regressor layers
        #for m in self.regressor.modules():
            #if type(m) == SeparableConvBlock:
                #fuse_modules(m, ['conv', 'bn'], inplace=True)

        # Fuse QAT_Classifier layers
        for m in self.classifier.modules():
            if type(m) == SeparableConvBlock:
                fuse_modules(m, ['depthwise_conv','pointwise_conv','bn','swish'], inplace=True)

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print('Ignoring ' + str(e) + '"')


#<---------HELPER FUNCTIONS----------->

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
ap.add_argument('--cuda', type=boolean_string, default=True)
ap.add_argument('--device', type=int, default=0)
ap.add_argument('--float16', type=boolean_string, default=False)
ap.add_argument('--override', type=boolean_string, default=True, help='override previous bbox results file if exists')
args = ap.parse_args()

compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'/content/QAT_ED_PyTorch/weights/efficientdet-d0.pth' if args.weights is None else args.weights

params = yaml.safe_load(open(f'/content/QAT_ED_PyTorch/projects/coco.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef], mean=params['mean'], std=params['std'])
        x = torch.from_numpy(framed_imgs[0])
        # if use_cuda:
        #   x = x.cuda()
        if use_float16:
          x = x.half()
        else:
          x = x.float()
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')
    
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
  
def load_model(model_file):
    params = Params(f'/content/QAT_ED_PyTorch/projects/coco.yml')
    model = EfficientDetBackbone(num_classes=len(params.obj_list),compound_coef=0,ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


#<---------DEFINE DATASET AND DATALOADERS----------->


def main():

    saved_model_dir = '/content/QAT_ED_PyTorch/weights'
    float_model_file = '/efficientdet-d0.pth'
    scripted_float_model_file = 'efficientdet_quantization_scripted.pth'
    scripted_quantized_model_file = 'efficientdet_quantization_scripted_quantized.pth'
    SET_NAME = params['val_set']
    VAL_GT = f'/content/QAT_ED_PyTorch/datasets/coco/annotations/instances_val2017.json'
    VAL_IMGS = f'/content/QAT_ED_PyTorch/datasets/coco/val2017/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir+float_model_file)
    # if use_cuda:
    #   float_model.cuda()    
    if use_float16:
      float_model.half()
    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.

    #print('\nBefore fusion: \n\n', float_model.features[1].conv)
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

    # Note fusion of Conv+BN+Relu and Conv+Relu
    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    # if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
    #   evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, float_model)
    

    # _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')

    #print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * 50, top1.avg))
    #torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

    num_calibration_batches = 32

    myModel = load_model(saved_model_dir + float_model_file)
    myModel.eval()

    # Fuse Conv, bn and relu
    myModel.fuse_model()

    # Specify quantization configuration
    # Start with simple min/max range estimation and per-tensor quantization of weights
    myModel.qconfig = torch.ao.quantization.default_qconfig
    print(myModel.qconfig)
    torch.ao.quantization.prepare(myModel, inplace=True)

    # Calibrate first
    #print('Post Training Quantization Prepare: Inserting Observers')
    #print('\n Inverted Residual Block:After observer insertion \n\n', myModel.features[1].conv)

    # Calibrate with the training set
    # if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
    #   evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, myModel)
    

    # _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
    # print('Post Training Quantization: Calibration done')

    # Convert to quantized model
    torch.ao.quantization.convert(myModel, inplace=True)
    #print('Post Training Quantization: Convert done')
    #print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',myModel.features[1].conv)

    print("Size of model after quantization")
    print_size_of_model(myModel)

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
      evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, myModel)
    

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')

    per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
    per_channel_quantized_model.eval()
    per_channel_quantized_model.fuse_model()
# The old 'fbgemm' is still available but 'x86' is the recommended default.
    per_channel_quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    print(per_channel_quantized_model.qconfig)

    torch.ao.quantization.prepare(per_channel_quantized_model, inplace=True)

    torch.ao.quantization.convert(per_channel_quantized_model, inplace=True)
    



if __name__ == "__main__":
    main()

    
