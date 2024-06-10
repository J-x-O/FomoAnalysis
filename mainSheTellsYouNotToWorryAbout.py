from tqdm import tqdm

from src.ExtractFrameData import clear_cache, extract_frame_data, cache_frame_data,  has_all_side_cars
from src.VideoUtil import find_all_videos


def successive_face_analysis():
    videos = find_all_videos("data/survey")
    todo = [v for v in videos if not has_all_side_cars(v)]
    for target in tqdm(todo, desc="Extracting"):
        data = extract_frame_data(target, console=False)
        cache_frame_data(target, data)


def nuke_all():
    videos = find_all_videos("data/survey")
    for video in tqdm(videos):
        clear_cache(video)


if __name__ == '__main__':
   successive_face_analysis()