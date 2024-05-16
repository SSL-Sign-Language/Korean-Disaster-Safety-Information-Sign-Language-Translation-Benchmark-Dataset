import os
import glob
import logging
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

from .processor import Processor

class KeypointExtractor(Processor):
    def __init__(self) -> None:
        super().__init__()
    
    def _read_video_frames(self, video_name: str) -> List[str]:
        frame_paths = glob.glob(f'{os.path.join(self.video_root, video_name)}/*.{self.args.extension}', recursive=True)
        frame_paths.sort()
        return frame_paths
                
    def _extract_keypoints(self, video_name: str) -> None:
        frame_paths = self._read_video_frames(video_name)
        num_frames: int = len(frame_paths)
        
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(static_image_mode=False, 
                                  model_complexity=2,
                                  min_detection_confidence=0.5, 
                                  min_tracking_confidence=0.5) as holistic_model:
            left_hand_keypoints: np.ndarray = np.zeros((1, num_frames, 33, 3)) # Create an empty keypoint numpy array
            right_hand_keypoints: np.ndarray = np.zeros((1, num_frames, 33, 3))
            body_keypoints: np.ndarray = np.zeros((1, num_frames, 33, 3))
            
            for time_step, frame_path in enumerate(frame_paths):
                image: Optional[np.ndarray] = self._process_frame(frame_path)
                if image is None:
                    continue

                results = holistic_model.process(image)
                self._update_keypoints(results, left_hand_keypoints, right_hand_keypoints, body_keypoints, image.shape, time_step)
            
            keypoints_concat: np.ndarray = np.concatenate([left_hand_keypoints, right_hand_keypoints, body_keypoints], axis=0)
            keypoints_transposed: np.ndarray = np.transpose(keypoints_concat, (3, 1, 2, 0))
            
            np.save(f'{self.npy_path}/{video_name}.npy', keypoints_transposed)

    def _process_frame(self, frame_path: str) -> Optional[np.ndarray]:
        try:
            image: np.ndarray = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            return None

    def _update_keypoints(self, results: Any, left_hand_keypoints: np.ndarray, right_hand_keypoints: np.ndarray, 
                          body_keypoints: np.ndarray, image_size: Tuple[int, int, int], time_step: int) -> None:
        image_height, image_width, _ = image_size
        for i in range(33): # 33 is body keypoints in the MediaPipe Holistic module
            self._update_body_keypoints(i, results, body_keypoints, image_width, image_height, time_step)
            if i <= 20: # 21 is hand keypoints in the MediaPipe Holistic module
                self._update_hand_keypoints(i, results, right_hand_keypoints, left_hand_keypoints, image_width, image_height, time_step)

    def _update_body_keypoints(self, i: int, results: Any, body_keypoints: np.ndarray, 
                               image_width: int, image_height: int, time_step: int) -> None:
        try:
            pose_keypoint = results.pose_landmarks.landmark[i]
            body_keypoints[:, time_step, i, :] = np.array([
                pose_keypoint.x * image_width,
                pose_keypoint.y * image_height,
                pose_keypoint.visibility # Confidence of body kepoints
            ])
        except:
            if time_step > 0:
                body_keypoints[:, time_step, i, :] = body_keypoints[:, time_step - 1, i, :] # Use keypoint from neighboring frames if no keypoints are detected

    def _update_hand_keypoints(self, i: int, results: Any, right_hand_keypoints: np.ndarray, 
                               left_hand_keypoints: np.ndarray, image_width: int, image_height: int, time_step: int) -> None:
        self._update_specific_hand_keypoints(i, results.right_hand_landmarks, right_hand_keypoints, image_width, image_height, time_step)
        self._update_specific_hand_keypoints(i, results.left_hand_landmarks, left_hand_keypoints, image_width, image_height, time_step)

    def _update_specific_hand_keypoints(self, i: int, hand_landmarks: Any, hand_keypoints: np.ndarray, 
                                        image_width: int, image_height: int, time_step: int) -> None:
        if hand_landmarks:
            try:
                hand_landmark = hand_landmarks.landmark[i]
                hand_keypoints[:, time_step, i, :] = np.array([
                    hand_landmark.x * image_width,
                    hand_landmark.y * image_height, 1 # Confidence of hand kepoints = 1 (No hand's visibility in this version)
                ])
            except:
                if time_step > 0:
                    hand_keypoints[:, time_step, i, :] = hand_keypoints[:, time_step - 1, i, :]

    def process(self) -> None:
        logging.info('Keypoint extraction in progress...')
        _, mode_folder, _ = self._get_data_mode_path(self.args.processing_type)
        
        self.video_root: str = os.path.join(self.args.save_path, mode_folder, 'Video')
        self.npy_path: str = os.path.join(self.args.save_path, mode_folder, 'Keypoint')

        video_names = os.listdir(self.video_root)
        logging.info(f'Num of Video: {len(video_names)}')

        existing_npys = os.listdir(self.npy_path)
        existing_npys = [npy.split('.')[0] for npy in existing_npys]
        
        video_names_to_process = list(set(video_names) - set(existing_npys)) # Do not run completed data 
        
        with Pool(processes=self.args.mp) as pool:
            list(tqdm(pool.imap(self._extract_keypoints, video_names_to_process), total=len(video_names_to_process)))
        logging.info('Keypoint Extraction Completed')
