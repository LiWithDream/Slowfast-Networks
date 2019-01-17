import numpy as np
import torch
import torch.nn as nn


def frame_downsample(x, alpha):
    #Downsample to 4(slow branch) or 32(fast branch) frames
    #Here applys a random downsample, which can be inplaced
    frame_ds = np.sort(np.random.randint(low = x.size(2), size = 4 * alpha))
    x = x[:, :, frame_ds, ]
    return x

def conv1x1x1(in_planes, out_planes, stride = 1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size = 1,
                     stride = (1, stride, stride),
                     bias = False)

def conv3x1x1(in_planes, out_planes, stride = 1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size = (3, 1, 1),
                     stride = stride,
                     padding = (1, 0, 0),
                     bias = False)

def conv1x3x3(in_planes, out_planes, stride = 1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size = (1, 3, 3),
                     stride = (1, stride, stride),
                     padding = (0, 1, 1),
                     bias = False)
                
def conv3x3x3(in_planes, out_planes, stride =1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size = 3,
                     stride = (1, stride, stride),
                     padding = 1,
                     bias = False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, slowfast, inplanes, planes, stride = 1, downsample = None):
        super(BasicBlock, self).__init__()
        if slowfast == 'slow':
            self.conv1 = conv1x3x3(inplanes, planes, stride)
        else:
            self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    #The same as the paper
    #the param *slowfast decide the kernel size of the conv1 in the block
    expansion = 4

    def __init__(self, slowfast, inplanes, planes, stride = 1, downsample = None):
        super(Bottleneck, self).__init__()
        if slowfast == 'slow':
            self.conv1 = conv1x1x1(inplanes, planes)
        else:
            self.conv1 = conv3x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv1x3x3(planes, planes, stride = stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * 4)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace = True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Slowfast_ResNet(nn.Module):
    
    def __init__(self,
                 block,
                 slowfast,
                 layers,
                 shortcut_type = 'B',
                 zero_init_residual = False):
        super(Slowfast_ResNet, self).__init__()
        self.slowfast = slowfast
        #conv1 is the ResNet-B in 'Bag of Tricks for Image Classification with Convolutional Neural Networks'
        if slowfast[0] == 'slow':
            self.inplanes = 64
            self.alpha = 1
            self.conv1 = nn.Sequential(conv1x3x3(3, 64, 2),
                                       conv1x3x3(64, 64),
                                       conv1x3x3(64, 64))
            self.bn1 = nn.BatchNorm3d(64)
        else:
            self.inplanes = 8
            self.alpha = 8
            self.conv1 = nn.Sequential(conv3x3x3(3, 8, 2),
                                       conv1x3x3(8, 8),
                                       conv3x3x3(8, 8))
            self.bn1 = nn.BatchNorm3d(8)

        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool3d(kernel_size = (1, 3, 3),
                                    stride = (1, 2, 2),
                                    padding = (0, 1, 1))

        self.layer1 = self._make_layer(block, 64, slowfast[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, slowfast[1], layers[1], shortcut_type, stride = 2)
        self.layer3 = self._make_layer(block, 256, slowfast[2], layers[2], shortcut_type, stride = 2)
        self.layer4 = self._make_layer(block, 512, slowfast[3], layers[3], shortcut_type, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, slowfast_block, blocks, shortcut_type, stride = 1):
        downsample = None
        planes = int(planes / self.alpha)
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = nn.Sequential(conv1x1x1(self.inplanes,
                                                     planes * block.expansion,
                                                     stride),
                                           nn.BatchNorm3d(planes * block.expansion))
            else:
                #the ResNet-D in 'Bag of Tricks for Image Classification with Convolutional Neural Networks'
                downsample = nn.Sequential(nn.AvgPool3d((1, stride, stride)),
                                           conv1x1x1(self.inplanes,
                                                     planes * block.expansion),
                                           nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(slowfast_block, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(slowfast_block, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = frame_downsample(x, self.alpha)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        return x


class Concat_slowfast(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes = 101):
        super(Concat_slowfast, self).__init__()
        if block == BasicBlock:
            self.fc = nn.Linear(512 + 64, num_classes)
        else:
            self.fc = nn.Linear(2048 + 256, num_classes)
            
        self.slow_branch = Slowfast_ResNet(block,
                                           ['slow', 'slow', 'fast', 'fast'],
                                           layers)
        self.fast_branch = Slowfast_ResNet(block,
                                           ['fast', 'fast', 'fast', 'fast'],
                                           layers)
    
    def forward(self, x):
        x1 = self.slow_branch(x)
        x2 = self.fast_branch(x)

        x = torch.cat((x1, x2), dim = 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = Concat_slowfast(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = Concat_slowfast(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Concat_slowfast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Concat_slowfast(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = Concat_slowfast(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model 


'''
#test the model

model = resnet18()
print (model)

a = torch.randn(1, 3, 64, 224, 224)
b = model(a)
print (b.shape)
'''
