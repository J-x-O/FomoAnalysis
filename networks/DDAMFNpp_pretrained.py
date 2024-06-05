import torch
from torch import nn

from networks.DDAM import DDAMNet
from networks.MixedFeatureNet import MixedFeatureNet


class UpdatedDDAMNet(DDAMNet):
    def __init__(self, num_class=7, num_head=2, pretrained_path='networks/DDAMFNpp/pretrained/MFN_msceleb.pth'):
        super(UpdatedDDAMNet, self).__init__(num_class=num_class, num_head=num_head, pretrained=False)
        if pretrained_path:
            # Load the pretrained MixedFeatureNet model
            mixed_feature_net = torch.load(pretrained_path, map_location='cpu')
            if isinstance(mixed_feature_net, MixedFeatureNet):
                self.features = nn.Sequential(*list(mixed_feature_net.children())[:-4])
            else:
                raise TypeError("Expected loaded model to be an instance of MixedFeatureNet")