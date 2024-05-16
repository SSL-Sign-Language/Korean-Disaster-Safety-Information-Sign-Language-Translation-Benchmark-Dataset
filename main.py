import logging
from src.sign_processor import SignProcessor

def main():
    logging.basicConfig(
    format = '%(asctime)s:%(levelname)s:%(message)s',
    datefmt = '%m/%d/%Y %I:%M:%S %p',
    level = logging.INFO)
    
    sign_processor = SignProcessor()
    sign_processor.start()

if __name__ == "__main__":
    main()
