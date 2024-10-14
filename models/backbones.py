from ast import Raise
from turtle import forward
from numpy.lib.function_base import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from thop import profile
# from torchviz import make_dot
from torchsummary import summary
import os

from copy import deepcopy
from .R2D import resnet50, resnet18, resnet18_new
from .R3D import generate_model
from .I3D import I3D_backbone
from .VGG import vgg16_backbone
from .mmbackbones2 import create_mm_backbones
from .mytransformer import set_attn_args, SaveOutput, MyTransformerEncoder, MyTransformerEncoderLayer
from tools.datasets.TITAN import NUM_CLS_ATOMIC, NUM_CLS_COMPLEX, NUM_CLS_COMMUNICATIVE, NUM_CLS_TRANSPORTING, NUM_CLS_AGE
from config import cktp_root

pretrained_path = '../work_dirs/models/c3d-pretrained.pth'

FLATTEN_DIM = {
    'R3D18': 25088,
    'R3D18_clean': 25088,
    'R3D50': 100352,
    'I3D': 100352,
    'I3D_clean': 100352,
    'C3D':32768,
    'C3D_clean':32768,
    'C3D_t4': 8192,
    'C3D_t4_clean': 8192,
    'C3D_full': 8192,
}

LAST_DIM = {
    'R3D18': 512,
    'R3D18_clean':512,
    'R3D50': 2048,
    'I3D': 1024,
    'I3D_clean': 1024,
    'C3D': 512,
    'C3D_clean': 512,
    'C3D_new': 512,
    'C3D_t4': 512,
    'C3D_t4_clean': 512,
    'C3D_full': 512,
    'deeplabv3_resnet50': 2048,
    'deeplabv3_resnet101': 2048,
    'transformerencoder1D': 64,
    'pedgraphconv': 64,
    'pedgraphconv_seg': 64,
    'pedgraphR2D50': 2048,
    'pedgraphR2D18': 512,
    'pedgraphR2D18_new': 512,
    'pedgraphflat': 4608,
}

BACKBONE_TO_OUTDIM = {
    'C3D': 512,
    'C3D_new': 512,
    'C3D_clean': 512,
    'R3D18': 512,
    'R3D18_clean': 512,
    'R3D18_new': 512,
    'R3D34': 512,
    'R3D34_clean': 512,
    'R3D34_new': 512,
    'R3D50': 2048,
    'R3D50_new': 2048,
    'R3D50_clean': 2048,
    'ircsn152': 2048,
    'poseC3D_pretrained': 512,
    'poseC3D': 512,
    'poseC3D_clean': 512,
    'lstm': 128
}

BACKBONE_TO_TENSOR_ORDER = {
    'C3D': 3,
    'C3D_new': 3,
    'C3D_clean': 3,
    'R3D18': 3,
    'R3D18_clean': 3,
    'R3D18_new': 3,
    'R3D34': 3,
    'R3D34_clean': 3,
    'R3D34_new': 3,
    'R3D50': 3,
    'R3D50_new': 3,
    'R3D50_clean': 3,
    'ircsn152': 3,
    'poseC3D_pretrained': 3,
    'poseC3D': 3,
    'poseC3D_clean': 3,
    'I3D': 3,
    'lstm': 1,
    'VGG16': 3,
}


class BackBones(nn.Module):
    def __init__(self, backbone_name,
                 num_classes=2,
                 use_atomic=0, 
                 use_complex=0, 
                 use_communicative=0, 
                 use_transporting=0, 
                 use_age=0,
                 use_cross=1,
                 pool='avg',
                 trainable_weights=0,
                 m_task_weights=0,
                 init_class_weights=None) -> None:
        super(BackBones, self).__init__()
        self.num_classes = num_classes
        self.use_atomic = use_atomic
        self.use_complex = use_complex
        self.use_communicative = use_communicative
        self.use_transporting = use_transporting
        self.use_age = use_age
        self.use_cross = use_cross
        self.pool = pool
        self.trainable_weights = trainable_weights
        self.init_class_weights = init_class_weights
        self.m_task_weights = m_task_weights

        self.backbone = create_backbone(backbone_name=backbone_name)

        if self.pool != 'flatten':
            feat_channel = LAST_DIM[backbone_name]
            self.final_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            feat_channel = FLATTEN_DIM[backbone_name]

        # last layers
        last_in_dim = feat_channel
        if self.use_atomic:
            self.atomic_layer = nn.Linear(feat_channel, NUM_CLS_ATOMIC, bias=False)
            if self.use_atomic == 2:
                last_in_dim += NUM_CLS_ATOMIC
        if self.use_complex:
            self.complex_layer = nn.Linear(feat_channel, NUM_CLS_COMPLEX, bias=False)
            if self.use_complex == 2:
                last_in_dim += NUM_CLS_COMPLEX
        if self.use_communicative:
            self.communicative_layer = nn.Linear(feat_channel, NUM_CLS_COMMUNICATIVE, bias=False)
            if self.use_communicative == 2:
                last_in_dim += NUM_CLS_COMMUNICATIVE
        if self.use_transporting:
            self.transporting_layer = nn.Linear(feat_channel, NUM_CLS_TRANSPORTING, bias=False)
            if self.use_transporting == 2:
                last_in_dim += NUM_CLS_TRANSPORTING
        if self.use_age:
            self.age_layer = nn.Linear(feat_channel, NUM_CLS_AGE)
            if self.use_age == 2:
                last_in_dim += NUM_CLS_AGE
        self.last_layer = nn.Linear(last_in_dim, self.num_classes)

        # create class weights
        if self.trainable_weights:
            self.class_weights = nn.Parameter(torch.tensor(self.init_class_weights['cross']), requires_grad=True)
            if use_atomic:
                self.atomic_weights = nn.Parameter(torch.tensor(self.init_class_weights['atomic']), requires_grad=True)
            if use_complex:
                self.complex_weights = nn.Parameter(torch.tensor(self.init_class_weights['complex']), requires_grad=True)
            if use_communicative:
                self.communicative_weights = nn.Parameter(torch.tensor(self.init_class_weights['communicative']), requires_grad=True)
            if use_transporting:
                self.transporting_weights = nn.Parameter(torch.tensor(self.init_class_weights['transporting']), requires_grad=True)
        
        # create task weights
        self.logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_atomic:
            self.atomic_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_complex:
            self.complex_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_communicative:
            self.communicative_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_transporting:
            self.transporting_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
        if self.use_age:
            self.age_logs2 = nn.Parameter(torch.rand(()), requires_grad=True)
    
    def forward(self, x):
        m_logits = {}
        x = x['img']
        x = self.backbone(x)
        if self.pool != 'flatten':
            x = self.final_pool(x)
        x = x.view(x.size(0), -1)

        _logits = [x]
        logits = {}
        if self.use_atomic:
            atomic_logits = self.atomic_layer(x)
            logits['atomic'] = atomic_logits
            if self.use_atomic == 2:
                _logits.append(atomic_logits)
        if self.use_complex:
            complex_logits = self.complex_layer(x)
            logits['complex'] = complex_logits
            if self.use_complex == 2:
                _logits.append(complex_logits)
        if self.use_communicative:
            communicative_logits = self.communicative_layer(x)
            logits['communicative'] = communicative_logits
            if self.use_communicative == 2:
                _logits.append(communicative_logits)
        if self.use_transporting:
            transporting_logits = self.transporting_layer(x)
            logits['transporting'] = transporting_logits
            if self.use_transporting == 2:
                _logits.append(transporting_logits)
        if self.use_age:
            age_logits = self.age_layer(x)
            logits['age'] = age_logits
            if self.use_age == 2:
                _logits.append(age_logits)
        if self.use_cross:
            final_logits = self.last_layer(torch.concat(_logits, dim=1))
            logits['final'] = final_logits
        
        return logits

class C3DDecoder(nn.Module):
    def __init__(self) -> None:  # 1, 8, 8 -> 16 224 224
        super(C3DDecoder, self).__init__()
        self.up_sample = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 0, 0)), # 1 6 6
            # nn.Upsample(size=(4, 7, 7)),  # 1 6 6 -> 1 7 7
            nn.ConstantPad3d((1, 0, 1, 0, 0, 0), 0),  # 1 6 6 -> 1 7 7
            nn.Upsample(scale_factor=(2, 2, 2)), # 2 14 14
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            
            nn.Upsample(scale_factor=(2, 2, 2)), # 4 28 28
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 8 56 56
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 16 112 112
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(1, 2, 2)), # 16 224 224
            nn.Conv3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

        self._init_weight()
    
    def forward(self, x):
        x = self.up_sample(x)
        # print(x.size())
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

class PoseC3DDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2)),  # 8 6 6 -> 8 12 12
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)), # 8 6 6 -> 8 6 6
            # nn.ConstantPad3d((1, 0, 1, 0, 0, 0), 0),  # 1 6 6 -> 1 7 7
            # nn.Upsample(scale_factor=(2, 2, 2)), # 2 14 14
            # nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            
            nn.Upsample(scale_factor=(1, 2, 2)), # 8 24 24
            nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),

            nn.Upsample(scale_factor=(2, 2, 2)), # 16 48 48
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.Conv3d(64, 17, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
        )

    def forward(self, x):
        return self.up_sample(x)

class RNNEncoder(nn.Module):
    def __init__(self, in_dim=4, h_dim=128, cell='lstm') -> None:
        super(RNNEncoder, self).__init__()
        if cell == 'lstm':
            self.encoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        else:
            raise NotImplementedError('Ilegal cell type')

    def forward(self, x):
        # x: B, T, C
        # print(x.size())
        self.encoder.flatten_parameters()
        _, (h, c) = self.encoder(x)  # h: 1, b, c  c: 1, b, c

        return h[0]

class CNN_RNNEncoder(nn.Module):
    def __init__(self, h_dim=128, cell='lstm', cnn_name='vgg16') -> None:
        super(CNN_RNNEncoder, self).__init__()
        self.cnn_backbone = create_backbone(cnn_name)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_dim = 512
        if cell == 'lstm':
            self.encoder = nn.LSTM(batch_first=True, input_size=in_dim, hidden_size=h_dim)
        else:
            raise NotImplementedError('Ilegal cell type')
        
    def forward(self, x):
        # x: B, C, T, H, W
        obs_len = x.size(2)
        featmaps = []
        for i in range(obs_len):
            f = self.cnn_backbone(x[:, :, i])
            f = self.pool(f)  # B, C, 1, 1
            f = f.view(f.size(0), f.size(1))  # B, C
            featmaps.append(f)
        featmaps = torch.stack(featmaps, dim=1)  # B, T, C
        _, (h, c) = self.encoder(featmaps)

        return h[0]

class SegC2D(nn.Module):
    def __init__(self):
        super(SegC2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3a = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        return x

class SegC3D(nn.Module):
    def __init__(self):
        super(SegC3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)

        return x

class SkeletonConv2D(nn.Module):
    def __init__(self):
        super(SkeletonConv2D, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=(1, 1), stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=(1, 1))
        # self.bn4 = nn.BatchNorm2d(128)
        self._initialize_weights()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        # x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)
        # print('C3D output', torch.mean(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class C3DPose(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, bn=True):
        super(C3DPose, self).__init__()

        self.conv1 = nn.Conv3d(17, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn2 = nn.BatchNorm3d(128)

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn3 = nn.BatchNorm3d(256)

        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.bn4 = nn.BatchNorm3d(512)

        self.conv5 = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.last_channel = 512
        self.__init_weight()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.relu(self.conv5(x))
        x = self.relu(self.conv5b(x))
        feat = self.pool5(x)

        return feat

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class C3D_backbone(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False, t_downsample='new'):
        super(C3D_backbone, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # t padding 1 --> 2
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 16 112 112 --> ?

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # t stride 2 --> 1

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        if t_downsample == 'ori':
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        if t_downsample == 't4':
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0))
        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        
        self.last_channel = 512
        self.__init_weight()

        self.pretrained_path = os.path.join(cktp_root, 'c3d-pretrained.pth')
        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        feat = self.pool5(x)

        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)
        #
        # logits = self.fc8(x)

        return feat

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias"
                        }

        p_dict = torch.load(self.pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)
        print('Total layer num : {}, update layer num: {}'.format(len(p_dict.keys()), len(s_dict.keys())))
        # print('Total layer: ', p_dict.keys())
        # print('Update layer: ', s_dict.keys())

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class C3D_full(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, last_dim, pretrained=False):
        super(C3D_full, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))  # 

        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, last_dim)

        # self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.pretrained_path = '../work_dirs/models/c3d-pretrained.pth'
        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        # import pdb;pdb.set_trace()
        logits = x.reshape(-1, 8192)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        # logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        # "classifier.0.weight": "fc6.weight",
                        # "classifier.0.bias": "fc6.bias",
                        # # fc7
                        # "classifier.3.weight": "fc7.weight",
                        # "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(self.pretrained_path)
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class BackboneOnly(nn.Module):
    def __init__(self, backbone_name):
        super(BackboneOnly, self).__init__()
        self.backbone = create_backbone(backbone_name)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        if 'C3D' or 'R3D' in backbone_name:
            self.fc = nn.Linear(512, 2)
        else:
            self.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Deeplabv3Backbone(nn.Module):
    def __init__(self, backbone, key='out') -> None:
        super().__init__()
        self.backbone = backbone
        self.key = key
    
    def forward(self, x):
        if len(x.size()) == 5:  # B C T H W
            x = x[:,:,-1]  # B C T H W -> B C H W
        x = self.backbone(x)
        return x[self.key]

class CustomTransformerEncoder1D(nn.Module):
    def __init__(self, in_dim=2, d_model=64, nhead=8, num_encoder_layers=1):
        super(CustomTransformerEncoder1D, self).__init__()
        self.in_dim = in_dim
        self.d_model = d_model
        self.conv1d = nn.Conv1d(self.in_dim, d_model, kernel_size=1)
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=d_model, 
        #                                nhead=nhead, 
        #                                dim_feedforward=d_model,
        #                                activation="gelu",
        #                                batch_first=True),
        #     num_layers=num_encoder_layers
        # )
        # # register hook
        # for n,m in self.transformer.named_modules():
        #     if isinstance(m, nn.MultiheadAttention):
        #         set_attn_args(m)
        #         m.register_forward_hook(self.get_attn)
        # self.attn_list = []
        self.transformer = MyTransformerEncoder(
            MyTransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=d_model,
                activation="gelu",
                batch_first=True
            ),
            num_layers=num_encoder_layers
            )
    
    def get_attn(self, module, input, output):
        # print(f'getting attn {len(output)}')
        self.attn_list.append(output[1].clone().detach())  # n_layer * ()
    
    def time_embedding(self, x):
        B, C, T = x.size()
        pos = torch.arange(T, device=x.device).float().unsqueeze(0).expand(B, T).unsqueeze(-1)  # B T 1
        embedding = torch.zeros(B, T, C, device=x.device)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, C, 2).to(x.device) / C
        )  # d/2,
        embedding[:, :, 0::2] = torch.sin(pos * div_term)
        embedding[:, :, 1::2] = torch.cos(pos * div_term)

        return embedding.permute(0, 2, 1)  # B d T

    def forward(self, x):
        '''
        x:   (B 2 obslen nj) or (B obslen 4/5)
        '''
        self.attn_list = []
        # sklt or social
        if len(x.size()) == 4:  # B 2 obslen nj --> # B 2 obslen*nj
            B, in_dim, obslen, nj = x.size()
            x = x.reshape(B, in_dim, obslen*nj)
        # traj
        elif len(x.size()) == 3:
            x = x.permute(0, 2, 1)  # B obslen 4 --> B 4 obslen

        x = self.conv1d(x)  # B C T

        x = x + self.time_embedding(x)
        x = x.permute(0, 2, 1)  # B T C
        x, self.attn_list = self.transformer(x)
        x = x.permute(0, 2, 1)  # B C T
        # try:
        #     print(f'attn shape: {self.attn_list[0].size()}')
        # except:
        #     import pdb;pdb.set_trace()
        return x
    

class PedGraphConv(nn.Module):
    def __init__(self, 
                 seg_only=False, 
                 n_seg=4,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ch1, self.ch2 = 32, 64
        self.seg_only = seg_only
        self.n_seg = n_seg
        i_ch = 3+self.n_seg if not seg_only else self.n_seg
        self.ctx_encoder0 = nn.Sequential(
            nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch1), 
            nn.SiLU(),
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch1), 
            nn.SiLU(),
            nn.Conv2d(self.ch1, self.ch2, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch2), 
            nn.SiLU()
            )
    
    def forward(self, x):
        '''
        x:  (T) B 4 H W
        '''
        # get the last frame
        if len(x.size()) == 5:  # B 4 T H W
            x = x[:,:,-1]  # B 4 T H W -> B 4 H W
        one_hot = F.one_hot(x[:, -1].long(), num_classes=self.n_seg).float() # B H W -> B H W n_seg
        one_hot = one_hot.permute(0, 3, 1, 2)  # B n_seg H W
        if self.seg_only:
            x = one_hot
        else:
            x = torch.cat([x[:,:3], one_hot], dim=1) # B 3+n_seg H W
        x = self.ctx_encoder0(x) # b 64 h w
        return x

class PedGraphFlat(nn.Module):
    def __init__(self, 
                 seg_only=False, 
                 n_seg=4,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ch1, self.ch2 = 32, 64
        self.seg_only = seg_only
        self.n_seg = n_seg
        i_ch = 3+self.n_seg if not seg_only else self.n_seg
        self.ctx_encoder0 = nn.Sequential(
            nn.Conv2d(i_ch, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch1), 
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(self.ch1, self.ch1, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch1), 
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(self.ch1, self.ch2, kernel_size=3, stride=1, padding=0, bias=False), 
            nn.BatchNorm2d(self.ch2), 
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            nn.Conv2d(self.ch2, self.ch1, kernel_size=1, stride=1, padding=0, bias=False), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            )
    
    def forward(self, x):
        '''
        x:  (T) B 4 H W
        '''
        # get the last frame
        if len(x.size()) == 5:  # B 4 T H W
            x = x[:,:,-1]  # B 4 T H W -> B 4 H W
        one_hot = F.one_hot(x[:, -1].long(), num_classes=self.n_seg).float() # B H W -> B H W n_seg
        one_hot = one_hot.permute(0, 3, 1, 2)  # B n_seg H W
        if self.seg_only:
            x = one_hot
        else:
            x = torch.cat([x[:,:3], one_hot], dim=1) # B 3+n_seg H W
        x = self.ctx_encoder0(x) # b 64 h w
        return x

class PedGraphR2D(nn.Module):
    def __init__(self, 
                 seg_only=False, 
                 n_seg=4,
                 pretrained=False,
                 n_layers=50,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seg_only = seg_only
        self.n_seg = n_seg
        i_ch = 3+self.n_seg if not seg_only else self.n_seg
        self.conv0 = nn.Conv2d(i_ch, 3, kernel_size=1, stride=1, padding=0, bias=False)
        if n_layers == 50:
            self.backbone = resnet50(pretrained=pretrained)
        elif n_layers == 18:
            self.backbone = resnet18(pretrained=pretrained)

    def forward(self, x):
        '''
        x:  (T) B 4 H W
        '''
        # get the last frame
        if len(x.size()) == 5:  # B 4 T H W
            x = x[:,:,-1]  # B 4 T H W -> B 4 H W
        one_hot = F.one_hot(x[:, -1].long(), num_classes=self.n_seg).float() # B H W -> B H W n_seg
        one_hot = one_hot.permute(0, 3, 1, 2)  # B n_seg H W
        if self.seg_only:
            x = one_hot
        else:
            x = torch.cat([x[:,:3], one_hot], dim=1) # B 3+n_seg H W
        x = self.conv0(x)
        x = self.backbone(x) # b 64 h w
        return x

class PedGraphR2D_new(nn.Module):
    def __init__(self, 
                 seg_only=False, 
                 n_seg=4,
                 pretrained=False,
                 n_layers=50,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seg_only = seg_only
        self.n_seg = n_seg
        i_ch = 3+self.n_seg if not seg_only else self.n_seg
        self.conv0 = nn.Conv2d(i_ch, 3, kernel_size=1, stride=1, padding=0, bias=False)
        if n_layers == 50:
            self.backbone = resnet50(pretrained=pretrained)
        elif n_layers == 18:
            self.backbone = resnet18_new(pretrained=pretrained)

    def forward(self, x):
        '''
        x:  (T) B 4 H W
        '''
        # get the last frame
        if len(x.size()) == 5:  # B 4 T H W
            x = x[:,:,-1]  # B 4 T H W -> B 4 H W
        one_hot = F.one_hot(x[:, -1].long(), num_classes=self.n_seg).float() # B H W -> B H W n_seg
        one_hot = one_hot.permute(0, 3, 1, 2)  # B n_seg H W
        if self.seg_only:
            x = one_hot
        else:
            x = torch.cat([x[:,:3], one_hot], dim=1) # B 3+n_seg H W
        x = self.conv0(x)
        x = self.backbone(x) # b 64 h w
        return x

def create_backbone(backbone_name, 
                    modality=None, 
                    lstm_h_dim=128, 
                    lstm_input_dim=4, 
                    last_dim=487,
                    args=None,
                    **kwargs):
    if backbone_name == 'C3D_new':  # 3, 16, 224, 224 -> 512, 1, 8, 8
        backbone = C3D_backbone(pretrained=True, t_downsample='new')
    elif backbone_name == 'C3D':
        backbone = C3D_backbone(pretrained=True, t_downsample='ori')
    elif backbone_name == 'C3D_clean':
        backbone = C3D_backbone(pretrained=False, t_downsample='ori')
    elif backbone_name == 'C3D_full':
        backbone = C3D_full(last_dim=last_dim, pretrained=True)
    elif backbone_name == 'C3D_t4':
        backbone = C3D_backbone(pretrained=True, t_downsample='t4')
    elif backbone_name == 'C3D_t4_clean':
        backbone = C3D_backbone(pretrained=False, t_downsample='t4')
    elif backbone_name == 'R3D18':  # 3, 16, 224, 224 -> 512, 1, 7, 7
        pretrained_path = '../work_dirs/models/r3d18_KM_200ep.pth'
        backbone = generate_model(18)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D18_clean':
        backbone = generate_model(18)
    # elif backbone_name == 'R3D34':
    #     backbone = _R3D.generate_model(34, pretrained=True)
    # elif backbone_name == 'R3D34_new':
    #     backbone = _R3D.generate_model(34, t_downsample=False)
    elif backbone_name == 'R3D50':  # 3, 16, 224, 224 -> 2048 1 7 7
        pretrained_path = '../work_dirs/models/r3d50_KMS_200ep.pth'
        backbone = generate_model(50)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D50_no_max':  # 3, 16, 224, 224 -> 2048 1 7 7
        pretrained_path = '../work_dirs/models/r3d50_KMS_200ep.pth'
        backbone = generate_model(50, no_max_pool=True)
        pretrained = torch.load(pretrained_path, map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'R3D50_clean':
        backbone = generate_model(50, pretrained=False)
    elif backbone_name == 'R3D50_new':
        pretrained_path = '../work_dirs/models/r3d50_KMS_200ep.pth'
        backbone = generate_model(50, t_downsample=False)
        pretrained = torch.load(pretrained_path, 
                                map_location='cpu')
        p_dict = pretrained['state_dict']  # keys必须完全一致
        for name in list(p_dict.keys()):
            if name[:2] == 'fc':
                p_dict.pop(name)
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'I3D':
        backbone = I3D_backbone()
        p_dict = torch.load('/home/y_feng/workspace6/work/ProtoPNet/work_dirs/models/i3d_model_rgb.pth')
        p_dict.pop('conv3d_0c_1x1.conv3d.weight')
        p_dict.pop('conv3d_0c_1x1.conv3d.bias')
        backbone.load_state_dict(p_dict)
    elif backbone_name == 'I3D_clean':
        backbone = I3D_backbone()
    elif backbone_name == 'SK':
        backbone = SkeletonConv2D()
    elif backbone_name == 'segC3D':
        backbone = SegC3D()
    elif backbone_name == 'segC2D':
        backbone = SegC2D()
    elif backbone_name == 'vgg16':
        backbone = vgg16_backbone(pretrained=True)
    elif backbone_name == 'cnn_lstm':
        backbone = CNN_RNNEncoder(h_dim=lstm_h_dim)
    elif backbone_name == 'lstm':
        backbone = RNNEncoder(h_dim=lstm_h_dim, in_dim=lstm_input_dim)
    elif backbone_name == 'C3Dpose':
        backbone = C3DPose()
    elif backbone_name == 'poseC3D':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3D_pretrained':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3D_clean':  # (17, 15/16, 48, 48) -> (512, 8, 6, 6)
        backbone = create_mm_backbones(backbone_name, pretrain=False)
    elif backbone_name == 'ircsn152':
        backbone = create_mm_backbones(backbone_name, pretrain=True)
    elif backbone_name == 'poseC3Ddecoder':
        backbone = PoseC3DDecoder()
    elif backbone_name == 'C3Ddecoder':
        backbone = C3DDecoder()
    elif backbone_name == 'deeplabv3_resnet50':
        model = torch.hub.load('pytorch/vision', 
                           'deeplabv3_resnet50', 
                           weights='DeepLabV3_ResNet50_Weights.DEFAULT',
                           )
        backbone = Deeplabv3Backbone(model.backbone, key='out')
    elif backbone_name == 'deeplabv3_resnet101':
        model = torch.hub.load('pytorch/vision', 
                           'deeplabv3_resnet101', 
                           weights='DeepLabV3_ResNet101_Weights.DEFAULT',
                           )
        backbone = Deeplabv3Backbone(model.backbone, key='out')
    elif backbone_name == 'transformerencoder1D':
        if modality == 'traj':
            in_dim = 4
        elif modality == 'sklt':
            in_dim = 2
        elif modality == 'social':
            if args.social_format == 'ori_traj':
                in_dim = 4
            elif args.social_format == 'rel_loc':
                in_dim = 5
        elif modality == 'ego':
            in_dim = 1
        backbone = CustomTransformerEncoder1D(in_dim=in_dim)
    elif backbone_name == 'pedgraphconv':
        backbone = PedGraphConv(seg_only=False,
                                n_seg=len(args.seg_cls.split(',')))
    elif backbone_name == 'pedgraphconv_seg':
        backbone = PedGraphConv(seg_only=True,
                                n_seg=len(args.seg_cls.split(',')))
    elif backbone_name == 'pedgraphR2D50':
        backbone = PedGraphR2D(seg_only=False,
                                 n_seg=len(args.seg_cls.split(',')),
                                 n_layers=50,
                                 pretrained=False)
    elif backbone_name == 'pedgraphR2D18':
        backbone = PedGraphR2D(seg_only=False,
                                 n_seg=len(args.seg_cls.split(',')),
                                 n_layers=18,
                                 pretrained=False)
    elif backbone_name == 'pedgraphR2D18_new':
        backbone = PedGraphR2D_new(seg_only=False,
                                 n_seg=len(args.seg_cls.split(',')),
                                 n_layers=18,
                                 pretrained=False)
    elif backbone_name == 'pedgraphflat':
        backbone = PedGraphFlat(seg_only=False,
                                n_seg=len(args.seg_cls.split(',')))
    else:
        raise ValueError(backbone_name)
    return backbone

def record_conv3d_info(model):
    kernel_size_list = []
    stride_list = []
    padding_list = []
    for name, m in model.named_modules():
        if (isinstance(m, nn.Conv3d) or isinstance(m, nn.MaxPool3d)) and 'downsample' not in name:
            kernel_size_list.append(m.kernel_size)
            stride_list.append(m.stride)
            padding_list.append(m.padding)
    return kernel_size_list, stride_list, padding_list

def record_conv2d_info(model):
    kernel_size_list = []
    stride_list = []
    padding_list = []
    for name, m in model.named_modules():
        if (isinstance(m, nn.Conv2d) or isinstance(m, nn.MaxPool2d)) and 'downsample' not in name:
            kernel_size_list.append(m.kernel_size)
            stride_list.append(m.stride)
            padding_list.append(m.padding)
    return kernel_size_list, stride_list, padding_list

def record_sp_conv3d_info_w(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-1])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-1])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-1])
    return [k_list, s_list, p_list]

def record_sp_conv3d_info_h(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-2])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-2])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-2])
    return [k_list, s_list, p_list]

def record_sp_conv2d_info_h(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-2])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-2])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-2])
    return [k_list, s_list, p_list]

def record_sp_conv2d_info_w(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[-1])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[-1])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[-1])
    return [k_list, s_list, p_list]

def record_t_conv3d_info(conv_info):
    k_list = []
    s_list = []
    p_list = []
    for i in range(len(conv_info[0])):
        cur_k = conv_info[0][i]
        cur_s = conv_info[1][i]
        cur_p = conv_info[2][i]
        k_list.append(cur_k if isinstance(cur_k, int) else cur_k[0])
        s_list.append(cur_s if isinstance(cur_s, int) else cur_s[0])
        p_list.append(cur_p if isinstance(cur_p, int) else cur_p[0])
    return [k_list, s_list, p_list]


if __name__ == "__main__":
    inputs = torch.ones(1, 3, 16, 224, 224)  # B, C, T, H, W
    # net = C3D_backbone(pretrained=True)
    # net = create_backbone('R3D50')
    # print(net)
    # # print(net)
    # # for m in net.modules():
    # #     print(m)
    # # for name, para in net.named_parameters():
    # #     print(name, ':')

    # summary(net, input_size=[(3, 16, 224, 224)], batch_size=1, device="cpu")

    # outputs = net.forward(inputs)
    # print(outputs.size())
    inputs = torch.rand(1, 7, 224, 224)
    net = PedGraphFlat(seg_only=False, n_seg=4)
    outputs = net.forward(inputs)
    print(outputs.size())
    import pdb;pdb.set_trace()