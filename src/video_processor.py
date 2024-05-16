import os
import glob
import json
import logging
import math
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop
from PIL import Image

from .processor import Processor

Image.ANTIALIAS = Image.LANCZOS

H, W = 1080, 1920

class VideoProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()
        
    def _get_start_end(self, data: Dict[str, Any], time_margin: int = 1) -> Tuple[int, int]:
        try:
            total = sum(data.values(), [])
            start, end = min(i['start'] for i in total), max(i['end'] for i in total)
            return max(math.floor(start) - time_margin, 0), math.ceil(end) + time_margin
        except KeyError:
            logging.error("JSON error")
            return 0, -1

    def _get_y_top(self, pose_keypoints: List[float], point_margin: float) -> int:
        top_position = min(keypoint[j] for keypoint in pose_keypoints for j in range(1, 75, 3))
        return max(int(top_position - ((H - top_position) * point_margin)), 0)

    def _get_json_data(self, json_path: str) -> Tuple[int, int, int]:
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)
        start, end = self._get_start_end(json_data['sign_script'], self.args.time_margin)
        y_top = self._get_y_top(json_data['landmarks']['pose_keypoints_2d'], self.args.point_margin)
        return start, end, y_top

    def _video_to_images(self, start: int, end: int, y_top: int, video_path: str, folder_path: str) -> None:
        clip = VideoFileClip(video_path).subclip(start, None if end == -1 else end) # Drop inactive frames
        cropped_clip = crop(clip, x1=(clip.w / 2) - ((clip.h - y_top) / 2), y1=y_top,
                            x2=(clip.w / 2) + ((clip.h - y_top) / 2), y2=clip.h).resize((self.args.resize, self.args.resize)) # Signer centered crop
        for i, frame in enumerate(cropped_clip.iter_frames()):
            Image.fromarray(frame).save(os.path.join(folder_path, f"frame_{int(i):04}.{self.args.extension}"))

    def _save_images_from_video(self, file_paths: Tuple[str, str]) -> None:
        _, mode_folder, _ = self._get_data_mode_path(self.args.processing_type)
        folder_path = os.path.join(self.args.save_path, mode_folder, 'Video', self._get_filename_without_extension(file_paths[1]))
        os.makedirs(folder_path, exist_ok=True)
        
        start, end, y_top = self._get_json_data(file_paths[0])
        self._video_to_images(start, end, y_top, file_paths[1], folder_path)

    def _process_files(self, json_paths: List[str], video_paths: List[str]) -> None:
        logging.info(f"Selecting videos with FPS between {self.args.min_fps} and {self.args.max_fps}...")
        with mp.Pool(processes=self.args.mp) as pool:
            video_paths = list(tqdm(pool.imap(self._filter_videos_by_fps, video_paths), total=len(video_paths)))
        
        video_paths = self._sort_by_filename(list(filter(None, video_paths)))
        video_names = self._get_filenames_without_extension(video_paths)
        
        json_paths = [json_path for json_path in json_paths if self._get_filename_without_extension(json_path) in video_names]
        json_paths = self._sort_by_filename(json_paths)
        
        file_pairs = list(zip(json_paths, video_paths)) # [(json, video), ...]
        
        logging.info("Saving images for data...")
        with mp.Pool(processes=self.args.mp) as pool:
            list(tqdm(pool.imap(self._save_images_from_video, file_pairs), total=len(file_pairs)))
    
    def process(self) -> None:
        logging.info('Video preprocessing in progress...')
        json_paths, video_paths = self._find_matching_files()
        self._process_files(json_paths, video_paths)
        logging.info('Video Process Completed')
