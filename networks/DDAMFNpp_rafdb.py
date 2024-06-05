import torch

from networks.DDAM import DDAMNet


class DDAMFNppRAFDB(DDAMNet):
    def __init__(self, num_class=7, num_head=2, pretrained_path='networks/DDAMFNpp/checkpoints_ver2/rafdb_epoch20_acc0.9204_bacc0.8617.pth'):
        super(DDAMFNppRAFDB, self).__init__(num_class=num_class, num_head=num_head, pretrained=False)
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()