import os
import glob
import json
import logging
import collections
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from .processor import Processor

class LanguageProcessor(Processor):
    def __init__(self) -> None:
        super().__init__()

    def _find_json_file_paths(self) -> Dict[str, str]:
        mode_folder, _, mode_json = self._get_data_mode_path(self.args.processing_type)
        json_directory = os.path.join(self.args.root_path, '01.데이터', mode_folder, '라벨링데이터', mode_json, '2.untact_morpheme')
        json_file_paths = glob.glob(f'{json_directory}/**/*.json', recursive=True)
        return {os.path.basename(path).split('.')[0]: path for path in json_file_paths}

    def _list_video_names_within_max_frame(self) -> List[str]:
        _, mode_folder, _ = self._get_data_mode_path(self.args.processing_type)
        videos_directory = os.path.join(self.args.save_path, mode_folder, 'Video')
        video_folders = [name for name in os.listdir(videos_directory) if os.path.isdir(os.path.join(videos_directory, name))]
        return [name for name in video_folders if len(os.listdir(os.path.join(videos_directory, name))) <= self.args.max_frame] # Select max frame or less from the entire video data

    def _create_gloss_sequence(self, sign_data: Dict[str, Any]) -> str:
        try:
            sign_script = sign_data['sign_script']
            gestures = sign_script['sign_gestures_both'] + sign_script['sign_gestures_strong'] + sign_script['sign_gestures_weak']
            ordered_gestures = sorted(gestures, key=lambda gesture: gesture['start']) # Integrating Gloss in order by time
            return " ".join(gesture['gloss_id'] for gesture in ordered_gestures)
        except KeyError:
            return ""

    def _process_json_file(self, json_file_path: str) -> Tuple[str, Dict[str, str]]:
        _, mode_folder, _ = self._get_data_mode_path(self.args.processing_type)

        with open(json_file_path, 'r', encoding='utf-8') as file:
            json_content = json.load(file)
        video_id = json_content['metadata']['id']
        video_base_dir = os.path.join(self.args.save_path, mode_folder, 'Video', video_id)
        return video_id, {
            'video_path' : video_base_dir,
            'keypoint_path' : os.path.join(self.args.save_path, mode_folder, 'Keypoint', f'{video_id}.npy'),                        
            "korean_text": json_content['korean_text'],
            "gloss_sequence": self._create_gloss_sequence(json_content),
            "frame" : len(os.listdir(video_base_dir)),
        }

    def _make_vocabulary(self, processed_data: Dict[str, Dict[str, str]]) -> List[str]:
        gloss_vocab = [data['gloss_sequence'].split() for video_id, data in processed_data.items()]
        count_dict = collections.Counter([item for sublist in gloss_vocab for item in sublist])
        return [gloss for gloss, _ in count_dict.most_common()]
        
    def _save_results(self, serialized_data: str, gloss_vocabulary: List[str]) -> None:
        _, mode_folder, _ = self._get_data_mode_path(self.args.processing_type)
        json_save_path = os.path.join(self.args.save_path, mode_folder, 'Language', f'{mode_folder}.json')
        with open(json_save_path, 'w', encoding='utf-8') as output_file:
            output_file.write(serialized_data)
        vocab_save_path = os.path.join(self.args.save_path, mode_folder, 'Language', f'{mode_folder}.vocab')
        with open(vocab_save_path, 'w', encoding='utf-8') as file:
            for gloss in gloss_vocabulary:
                file.write(gloss + '\n')
                
    def process(self) -> None:
        logging.info('Language preprocessing in progress...')
        json_file_mapping = self._find_json_file_paths()
        video_names = self._list_video_names_within_max_frame()
        filtered_json_paths = [json_file_mapping.get(name) for name in video_names if name in json_file_mapping]

        with mp.Pool(processes=self.args.mp) as pool:
            processed_data = list(tqdm(pool.imap(self._process_json_file, filtered_json_paths), total=len(filtered_json_paths)))

        final_result = {video_id: data_row for video_id, data_row in processed_data}
        gloss_vocabulary = self._make_vocabulary(final_result)
        serialized_data = json.dumps(final_result, indent=4, ensure_ascii=False)

        self._save_results(serialized_data, gloss_vocabulary)
        logging.info('Language Process Completed')

