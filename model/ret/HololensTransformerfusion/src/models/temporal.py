import torch.nn as nn
from models.backbones.resnet import resnet18
from models.backbones.vgg import vgg16, vgg16_bn
from models.backbones import c3d
from models.backbones.r3d import r3d_18, r2plus1d_18
from models.attention import EncoderSelfAttention

backbone_dict = {'resnet': resnet18,
                 'vgg': vgg16, 'vgg_bn': vgg16_bn,
                 'c3d': c3d,
                 'r3d': r3d_18, 'r2plus1d': r2plus1d_18}

class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, configer, backbone: nn.Module, rgb_in_planes: int, depth_in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone=0.1,
                 **kwargs):
        super(_GestureTransformer, self).__init__()

        self.configer = configer
        self.rgb_in_planes = rgb_in_planes
        self.depth_in_planes = depth_in_planes
        self.depth_backbone = backbone(pretrained, depth_in_planes, dropout=dropout_backbone)
        self.rgb_backbone = backbone(pretrained, rgb_in_planes, dropout=dropout_backbone)

        self.self_attention = EncoderSelfAttention(512, 64, 64, **kwargs)

        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        self.classifier = nn.Linear(512, out_planes)


    def forward(self, rgb, depth):

        rgb_shape = rgb.shape
        rgb = rgb.view(-1, self.rgb_in_planes, rgb.shape[-2], rgb.shape[-1])
        rgb = self.rgb_backbone(rgb)
        rgb = rgb.view(rgb_shape[0], rgb_shape[1] // self.rgb_in_planes, -1)
        rgb = self.self_attention(rgb)
        rgb_logit = self.pool(rgb).squeeze(dim=1)
        rgb = self.classifier(rgb_logit)

        depth_shape = depth.shape

        depth = depth.view(-1, self.depth_in_planes, depth.shape[-2], depth.shape[-1])

        depth = self.depth_backbone(depth)
        depth = depth.view(depth_shape[0], depth_shape[1] // self.depth_in_planes, -1)

        depth = self.self_attention(depth)

        depth_logit = self.pool(depth).squeeze(dim=1)
        depth = self.classifier(depth_logit)

        if self.configer.get("fusionkind")=='late_fusion':
            logits = depth+rgb
        elif self.configer.get("fusionkind")=='feature_fusion':
            logits = depth_logit + rgb_logit
            logits = self.classifier(logits)

        return logits

def GestureTransoformer(configer, backbone: str="resnet", rgb_in_planes: int=3, depth_in_planes: int=1, n_classes: int=25, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformer(configer, backbone_dict[backbone], rgb_in_planes, depth_in_planes, n_classes, **kwargs)
    return model