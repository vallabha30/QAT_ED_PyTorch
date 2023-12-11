#<---------IMPORT STMTS----------->

import os
import sys
import time
import datetime
import numpy as np
from torch.serialization import MAP_LOCATION
from tqdm.autonotebook import tqdm
import argparse
import traceback

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import yaml
import json
from utils.sync_batchnorm import patch_replication_callback


from torch import nn
import torch.nn.functional as F
from efficientdet.loss import FocalLoss

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
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater, custom_collate
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
from efficientdet.model import BiFPN, Regressor, QAT_Classifier, EfficientNet
from efficientdet.utils import Anchors
from efficientnet.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding, Bn2dWrapper

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
        # Q: whether separate convshare bias between depthwise_conv and pointwise_conv or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.
        self.quant=QuantStub()
        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels, kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)
        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = Bn2dWrapper(num_features=out_channels, momentum=0.01, eps=1e-3)
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
        # # Attach observers to specific layers in your model
        self.bn1 = Bn2dWrapper(64)  # Replace with actual batch norm layer
        self.bn2 = Bn2dWrapper(128)  # Replace with actual batch norm layer
        # # Add more observers for other layers as needed

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, Bn2dWrapper):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]
        _, p3, p4, p5 = self.backbone_net(inputs)
        features = (p3, p4, p5)
        features = self.bifpn(features)
        regression = self.regressor(features)
        classification = self.classifier(features)
        anchors = self.anchors(inputs, inputs.dtype)
        # features=self.quant(features)
        # features=self.dequant(features)

        return features , regression, classification, anchors
              
    def fuse_model(self, is_qat=False):
        fuse_modules = torch.ao.quantization.fuse_modules_qat if is_qat else torch.ao.quantization.fuse_modules
        # Fuse BiFPN layers
        for m in self.bifpn.modules():
            if type(m) == SeparableConvBlock:
                fuse_modules(m, ['depthwise_conv','pointwise_conv','bn','swish'], inplace=True)      
        
        # Fuse Regressor layers
        for m in self.regressor.modules():
            if type(m) == SeparableConvBlock:
                fuse_modules(m, ['conv', 'bn','swish'], inplace=True)

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
class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(model, criterion, data_loader, neval_batches):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            features , regression, classification, anchors = model(image)
            loss = criterion(classification, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            print('.', end = '')
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
            if cnt >= neval_batches:
                 return top1, top5

    return top1, top5


def load_model(model_file):
    params = Params(f'/content/QAT_ED_PyTorch/projects/coco.yml')
    model = EfficientDetBackbone(num_classes=len(params.obj_list),compound_coef=0,ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')),strict=False)

    model.requires_grad_(False)
    return model

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
      model.train()
      top1 = AverageMeter('Acc@1', ':6.2f')
      top5 = AverageMeter('Acc@5', ':6.2f')
      avgloss = AverageMeter('Loss', '1.5f')

      cnt = 0
      for image, target in data_loader:
          start_time = time.time()
          print('.', end = '')
          cnt += 1
          image, target = image.to(device), target.to(device)
          output = model(image)
          loss = criterion(output, target)
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          acc1, acc5 = accuracy(output, target, topk=(1, 5))
          top1.update(acc1[0], image.size(0))
          top5.update(acc5[0], image.size(0))
          avgloss.update(loss, image.size(0))
          if cnt >= ntrain_batches:
              print('Loss', avgloss.avg)

              print('Training: * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))
              return

      print('Full imagenet train set:  * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
            .format(top1=top1, top5=top5))
      return

def run_benchmark(model_file, img_loader):
    elapsed = 0
    model = torch.jit.load(model_file)
    model.eval()
    num_batches = 5
    # Run the scripted model on a few batches of images
    for i, (images, target) in enumerate(img_loader):
        if i < num_batches:
            start = time.time()
            output = model(images)
            end = time.time()
            elapsed = elapsed + (end-start)
        else:
            break
    num_images = images.size()[0] * num_batches

    print('Elapsed time: %3.0f ms' % (elapsed/num_images*1000))
    return elapsed

#<--------- DATALOADERS ------------>

def prepare_data_loaders(data_path):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    dataset = torchvision.datasets.CocoDetection(
        root=os.path.join(data_path,"train2017"), annFile="/content/QAT_ED_PyTorch/datasets/coco/annotations/instances_train2017.json", 
        transform=transforms.Compose([
            transforms.Resize(512),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    dataset_test = torchvision.datasets.CocoDetection(
        root=os.path.join(data_path,"val2017"), annFile="/content/QAT_ED_PyTorch/datasets/coco/annotations/instances_val2017.json",
        transform=transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_batch_size = 30
    eval_batch_size = 50

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,collate_fn=custom_collate,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=eval_batch_size,collate_fn=custom_collate,
        sampler=test_sampler)

    return data_loader, data_loader_test
#<---------DEFINE DATASET----------->


def main():

    data_path = '/content/QAT_ED_PyTorch/datasets/coco/'
    saved_model_dir = '/content/QAT_ED_PyTorch/weights/'
    float_model_file = 'efficientdet-d0_updated'
    scripted_float_model_file = 'efficientdet_d0_quantization_scripted.pth'
    scripted_quantized_model_file = 'efficientdet_d0_quantization_scripted_quantized.pth'

    train_batch_size = 30
    eval_batch_size = 50

    data_loader, data_loader_test = prepare_data_loaders(data_path)
    criterion = nn.CrossEntropyLoss()
    float_model = load_model(saved_model_dir + float_model_file).to('cpu')

    # Next, we'll "fuse modules"; this can both make the model faster by saving on memory access
    # while also improving numerical accuracy. While this can be used with any model, this is
    # especially common with quantized models.
    float_model.eval()

    # Fuses modules
    float_model.fuse_model()

#<---------------------- BASELINE ACCURACY -------------------------->

    num_eval_batches = 1000

    print("Size of baseline model")
    print_size_of_model(float_model)

    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=num_eval_batches)
    print('Evaluation accuracy on %d images, %2.2f'%(num_eval_batches * eval_batch_size, top1.avg))
    torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

#<--------------------------QAT--------------------------------------->
    qat_model = load_model(saved_model_dir + float_model_file)
    qat_model.fuse_model(is_qat=True)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr = 0.0001)
    # The old 'fbgemm' is still available but 'x86' is the recommended default.
    qat_model.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')

    torch.ao.quantization.prepare_qat(qat_model, inplace=True)

    num_train_batches = 20

# QAT takes time and one needs to train over a few epochs.
# Train and check accuracy after each epoch
    for nepoch in range(8):
        train_one_epoch(qat_model, criterion, optimizer, data_loader, torch.device('cpu'), num_train_batches)
        if nepoch > 3:
            # Freeze quantizer parameters
            qat_model.apply(torch.ao.quantization.disable_observer)
        if nepoch > 2:
            # Freeze batch norm mean and variance estimates
            qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        # Check the accuracy after each epoch
        quantized_model = torch.ao.quantization.convert(qat_model.eval(), inplace=False)
        quantized_model.eval()
        top1, top5 = evaluate(quantized_model,criterion, data_loader_test, neval_batches=num_eval_batches)
        print('Epoch %d :Evaluation accuracy on %d images, %2.2f'%(nepoch, num_eval_batches * eval_batch_size, top1.avg))
    
    run_benchmark(saved_model_dir + scripted_float_model_file, data_loader_test)

    run_benchmark(saved_model_dir + scripted_quantized_model_file, data_loader_test)

if __name__ == "__main__":
    main()
