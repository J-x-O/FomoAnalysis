import torch

from networks.DDAMFNpp_affectnet7 import DDAMFNppAffectnet7
from networks.DDAMFNpp_rafdb import DDAMFNppRAFDB
from src.FrameData import has_all_side_cars, ExtractedEmotionSet, load_frame_data, load_single_frame_data
from src.RetinaFaceAlign import transform_stack
from src.VideoUtil import VideoTarget, FrameIterator


models = {
    "affectnet7": DDAMFNppAffectnet7(),
    "rafdb": DDAMFNppRAFDB()
}

def get_missing_models(video: VideoTarget):
    return {k: v for k, v in models.items() if not video.has_side_car(f"{k}.json")}

def get_existing_models(video: VideoTarget):
    return {k: v for k, v in models.items() if video.has_side_car(f"{k}.json")}

def extract_frame_data(video: VideoTarget, skipp_existing: bool = False, console: bool = True) -> ExtractedEmotionSet:
    if skipp_existing and has_all_side_cars(video):
        print(f"Skipping {video.video_title} as all side cars are present.")
        return load_frame_data(video)

    data = { }

    model_selection = models if not skipp_existing else get_missing_models(video)
    for model_name in model_selection.keys():
        data[model_name] = []

    for frame_index, frame in FrameIterator(video, console=console):
        cropped = transform_stack(frame)
        for model_name, model in model_selection.items():
            with torch.no_grad():
                outputs, _, _ = model(cropped)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().tolist()
            data[model_name].append(probabilities[0])

    if not skipp_existing: return data

    for model_name in get_existing_models(video).keys():
        data[model_name] = load_single_frame_data(video, model_name)
    return data

