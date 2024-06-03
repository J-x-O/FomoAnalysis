import torch
from data import cfg_mnet, cfg_re50
import cv2
import numpy as np
from skimage import transform as trans
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from face_align_v2 import src_map

def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M, min_index, min_error = [], [], float('inf')
    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T).T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        if error < min_error:
            min_error, min_M, min_index = error, M, i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112):
    M, _ = estimate_norm(landmark, image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)

def load_model(model, pretrained_path, load_to_cpu):
    pretrained_dict = torch.load(pretrained_path, map_location='cpu' if load_to_cpu else lambda storage, loc: storage.cuda(torch.cuda.current_device()))
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def align_faces(image_path, model_path, device='cuda'):
    # Load the RetinaFace model
    cfg = cfg_mnet if 'mobile0.25' in model_path else cfg_re50
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, model_path, device == 'cpu')
    net = net.to(device).eval()

    # Read image
    img_raw = cv2.imread(image_path)
    img = np.float32(img_raw)
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Forward pass
    loc, conf, landms = net(img)
    priorbox = PriorBox(cfg, image_size=(img.shape[2], img.shape[3]))
    prior_data = priorbox.forward().to(device).data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance']).cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    inds = np.where(scores > 0.6)[0]
    boxes = boxes[inds]
    scores = scores[inds]
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, 0.35)
    dets = dets[keep, :]
    for b in dets:
        if b[4] < 0.6:
            continue
        b = list(map(int, b))
        face_landmarks = np.array([[b[5], b[6]], [b[7], b[8]], [b[9], b[10]], [b[11], b[12]], [b[13], b[14]]], dtype=np.float32)
        aligned_face = norm_crop(img_raw, face_landmarks)
        return aligned_face
