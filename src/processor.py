from .args import get_args
from typing import Any, Dict, List, Optional, Tuple

class Processor:
    def __init__(self) -> None:
        args = get_args()
        self.args = args
        
    def _get_data_mode_path(self, data_mode: str) -> Optional[Tuple[str, str]]:
        data_mode_paths = {'train': ('1.Training', 'Train', '03_JSON_TrL'), 
                           'valid': ('2.Validation', 'Validation', '03_JSON_VL')}
        return data_mode_paths.get(data_mode)