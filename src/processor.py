import os
import glob
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from .args import get_args

class Processor:
    def __init__(self) -> None:
        args = get_args()
        self.args = args
        
    def _filter_videos_by_fps(self, video_path: str) -> Optional[str]:
        clip = VideoFileClip(video_path)
        return video_path if self.args.min_fps <= clip.fps <= self.args.max_fps else None
    
    def _sort_by_filename(self, paths: List[str]) -> List[str]:
        return sorted(paths, key=self._get_filename_without_extension)
    
    def _get_filename_without_extension(self, path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    def _get_filenames_without_extension(self, paths: List[str]) -> set:
        return {self._get_filename_without_extension(p) for p in paths}
        
    def _find_file_paths(self, file_type: str) -> List[str]:
        assert file_type in ['json', 'mp4'], 'file_type must be either "json" or "mp4"'
        
        mode_folder, _, mode_json = self._get_data_mode_path(self.args.processing_type)
        file_directory = os.path.join(self.args.root_path, '01.데이터', mode_folder,
                                      '라벨링데이터' if file_type == 'json' else '원천데이터',
                                      mode_json if file_type == 'json' else '1.mp4',
                                      '2.untact_morpheme' if file_type == 'json' else '3.crowd')
        return glob.glob(f'{file_directory}/**/*.{file_type}', recursive=True)    
    
    def _find_json_file_paths(self) -> Dict[str, str]:
        mode_folder, _, mode_json = self._get_data_mode_path(self.args.processing_type)
        json_directory = os.path.join(self.args.root_path, '01.데이터', mode_folder, '라벨링데이터', mode_json, '2.untact_morpheme')
        json_file_paths = glob.glob(f'{json_directory}/**/*.json', recursive=True)
        return {os.path.basename(path).split('.')[0]: path for path in json_file_paths}
    
    def _read_video_frames(self, video_name: str) -> List[str]:
        frame_paths = glob.glob(f'{os.path.join(self.video_root, video_name)}/*.{self.args.extension}', recursive=True)
        frame_paths.sort()
        return frame_paths
    
    def _find_matching_files(self) -> Tuple[List[str], List[str]]:
        json_paths, video_paths = map(sorted, (self._find_file_paths(ft) for ft in ('json', 'mp4')))
        
        json_names, video_names = (self._get_filenames_without_extension(paths) for paths in (json_paths, video_paths))
        matching_files = json_names & video_names # intersection of json and video
                
        return [p for p in json_paths if self._get_filename_without_extension(p) in matching_files], \
               [p for p in video_paths if self._get_filename_without_extension(p) in matching_files]
    
    def _process_frame(self, frame_path: str) -> Optional[np.ndarray]:
        try:
            image: np.ndarray = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            return None
    
    def _get_data_mode_path(self, data_mode: str) -> Optional[Tuple[str, str]]:
        data_mode_paths = {'train': ('1.Training', 'Train', '03_JSON_TrL'), 
                           'valid': ('2.Validation', 'Validation', '03_JSON_VL')}
        return data_mode_paths.get(data_mode)