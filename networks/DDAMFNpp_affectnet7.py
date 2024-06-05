import torch

from networks.DDAM import DDAMNet


class DDAMFNppAffectnet7(DDAMNet):
    def __init__(self, num_class=7, num_head=2, pretrained_path='networks/DDAMFNpp/checkpoints_ver2/affecnet7_epoch19_acc0.671.pth'):
        super(DDAMFNppAffectnet7, self).__init__(num_class=num_class, num_head=num_head, pretrained=False)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()