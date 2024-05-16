import os

from .keypoint_extractor import KeypointExtractor
from .language_processor import LanguageProcessor
from .video_processor import VideoProcessor

class SignProcessor:
    def __init__(self) -> None:
        self.video_processor = VideoProcessor()
        self.keypoint_extractor = KeypointExtractor()
        self.language_processor = LanguageProcessor()

    def _make_dir_if_not_exists(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    def _create_subfolders(self, parent_dir: str, subfolders: list) -> None:
        for folder in subfolders:
            self._make_dir_if_not_exists(os.path.join(parent_dir, folder))

    def _prepare_directory_structure(self, save_path: str) -> None:
        self._make_dir_if_not_exists(save_path)

        # Prepare the Train & Validation directory
        train_dir = os.path.join(save_path, "Train")
        validation_dir = os.path.join(save_path, "Validation")
        self._make_dir_if_not_exists(train_dir)
        self._make_dir_if_not_exists(validation_dir)

        # Create a subfolder within Train & Validation
        subfolders = ["Keypoint", "Language", "Video"]
        self._create_subfolders(train_dir, subfolders)
        self._create_subfolders(validation_dir, subfolders)

    def start(self) -> None:
        self._prepare_directory_structure(self.video_processor.args.save_path) # make folder for result
        self.video_processor.process()
        self.keypoint_extractor.process()
        self.language_processor.process()
