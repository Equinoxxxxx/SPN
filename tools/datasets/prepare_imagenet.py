import os
import requests
import tarfile
from tqdm import tqdm
import wget
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageNetDownloader:
    def __init__(self, download_dir="imagenet_data"):
        self.base_url = "https://image-net.org/data/"
        self.download_dir = download_dir
        
    def download_file(self, url, destination):
        try:
            logger.info(f"Downloading {url} to {destination}")
            wget.download(url, destination)
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            
    def download_dataset(self, full_dataset=False):
        """
        Note: You'll need to register at image-net.org and replace these URLs
        with your actual download URLs that contain your authentication tokens
        """
        os.makedirs(self.download_dir, exist_ok=True)
        
        if full_dataset:
            # URLs for full dataset (you'll need to get these from ImageNet)
            urls = [
                "FULL_DATASET_URL_HERE",
                "FULL_ANNOTATIONS_URL_HERE"
            ]
        else:
            # URLs for ILSVRC subset
            urls = [
                "ILSVRC_SUBSET_URL_HERE",
                "ILSVRC_ANNOTATIONS_URL_HERE"
            ]
            
        for url in urls:
            filename = url.split('/')[-1]
            destination = os.path.join(self.download_dir, filename)
            self.download_file(url, destination)
            
        logger.info("Download completed!")

if __name__ == "__main__":
    downloader = ImageNetDownloader(download_dir="imagenet_data")
    downloader.download_dataset(full_dataset=False)  # Set to True for full dataset