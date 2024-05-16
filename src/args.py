import logging
import argparse
import multiprocessing as mp

# Default paths
# DEFAULT_DATA_ROOT_PATH = "/home/sign/coling_2024/114.재난_안전_정보_전달을_위한_수어영상_데이터"
DEFAULT_SAVE_PATH = './result'

def create_parser():
    """
    Creates an argparse instance to handle command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Process command-line arguments for dataset preparation.')
    
    # Arguments for data and save paths
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path of the unzipped folder downloaded from AIHUB.') 
    parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH,
                        help='Path where the processed data will be saved.')
    
    # Arguments for processing type and frame settings
    parser.add_argument('--processing_type', type=str, choices=['train', 'valid'], default='train',
                        help='Specifies the processing type: train or valid.')
    parser.add_argument('--max_frame', type=int, default=400,
                        help='Maximum number of frames in the video to be filtered.')
    
    # Arguments for time and point margin settings
    parser.add_argument('--time_margin', type=int, default=1,
                        help='Margin (in seconds) for timestamp during gloss time mapping to drop inactive frames.')
    parser.add_argument('--point_margin', type=float, default=0.2,
                        help='Margin (in ratio) for signer positioning in cropped frames, affecting background inclusion and signer truncation risk.')

    # Arguments for video FPS settings
    parser.add_argument('--min_fps', type=int, default=25,
                        help='Minimum FPS of videos to be filtered.')
    parser.add_argument('--max_fps', type=int, default=30,
                        help='Maximum FPS of videos to be filtered.')
    
    # Arguments for image extension and resize settings
    parser.add_argument('--extension', type=str, choices=['jpg', 'png'], default='jpg',
                        help='Image file extension (jpg or png).')
    parser.add_argument('--resize', type=int, default=256,
                        help='Dimension to which images will be resized (in pixels).')
    
    # Argument for multiprocessing thread count
    parser.add_argument('--mp', type=int, default=mp.cpu_count(),
                        help='Number of threads for multiprocessing.')
    
    return parser

def get_args():
    """
    Parses and returns the command-line arguments using the created parser.
    """
    parser = create_parser()
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    logging.info(args)
