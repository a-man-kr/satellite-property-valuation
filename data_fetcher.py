"""
Satellite Image Fetcher for Property Valuation
Downloads satellite/aerial imagery using lat/long coordinates from property data.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()


class SatelliteImageFetcher:
    """Fetch satellite images from various APIs."""
    
    def __init__(self, api_provider: str = "mapbox", output_dir: str = "data/images"):
        self.api_provider = api_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # API configuration
        self.mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")
        self.google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        # Image settings
        self.zoom = 18  # High zoom for property-level detail
        self.image_size = 256  # 256x256 pixels
        
    def _get_mapbox_url(self, lat: float, lon: float) -> str:
        """Generate Mapbox Static Images API URL."""
        return (
            f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
            f"{lon},{lat},{self.zoom},0/{self.image_size}x{self.image_size}"
            f"?access_token={self.mapbox_token}"
        )
    
    def _get_google_url(self, lat: float, lon: float) -> str:
        """Generate Google Maps Static API URL."""
        return (
            f"https://maps.googleapis.com/maps/api/staticmap?"
            f"center={lat},{lon}&zoom={self.zoom}&size={self.image_size}x{self.image_size}"
            f"&maptype=satellite&key={self.google_api_key}"
        )
    
    def _get_osm_url(self, lat: float, lon: float) -> str:
        """Generate OpenStreetMap tile URL (free, no API key needed)."""
        import math
        n = 2 ** self.zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
        return f"https://tile.openstreetmap.org/{self.zoom}/{x}/{y}.png"
    
    def fetch_image(self, property_id: str, lat: float, lon: float) -> bool:
        """Download satellite image for a single property."""
        output_path = self.output_dir / f"{property_id}.png"
        
        # Skip if already downloaded
        if output_path.exists():
            return True
        
        # Get URL based on provider
        if self.api_provider == "mapbox":
            url = self._get_mapbox_url(lat, lon)
        elif self.api_provider == "google":
            url = self._get_google_url(lat, lon)
        else:
            url = self._get_osm_url(lat, lon)
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            return True
            
        except Exception as e:
            print(f"Error fetching image for {property_id}: {e}")
            return False
    
    def fetch_all_images(self, df: pd.DataFrame, max_workers: int = 5, 
                         rate_limit: float = 0.1) -> dict:
        """
        Download satellite images for all properties in dataframe.
        
        Args:
            df: DataFrame with 'id', 'lat', 'long' columns
            max_workers: Number of parallel download threads
            rate_limit: Seconds to wait between requests
        
        Returns:
            Dictionary with download statistics
        """
        results = {"success": 0, "failed": 0, "skipped": 0}
        
        print(f"Fetching {len(df)} satellite images using {self.api_provider}...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            property_id = str(row['id'])
            lat = row['lat']
            lon = row['long']
            
            output_path = self.output_dir / f"{property_id}.png"
            if output_path.exists():
                results["skipped"] += 1
                continue
            
            success = self.fetch_image(property_id, lat, lon)
            if success:
                results["success"] += 1
            else:
                results["failed"] += 1
            
            time.sleep(rate_limit)  # Rate limiting
        
        print(f"\nDownload complete: {results}")
        return results


def create_placeholder_images(df: pd.DataFrame, output_dir: str = "data/images"):
    """
    Create placeholder images for testing when API is not available.
    Generates simple colored images based on property features.
    """
    import numpy as np
    from PIL import Image
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Creating placeholder images for testing...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        property_id = str(row['id'])
        img_path = output_path / f"{property_id}.png"
        
        if img_path.exists():
            continue
        
        # Create a simple gradient image based on location
        lat_norm = (row['lat'] - 47.0) / 1.0  # Normalize latitude
        lon_norm = (row['long'] + 122.5) / 1.0  # Normalize longitude
        
        # Generate RGB values based on location and features
        r = int(np.clip(lat_norm * 255, 0, 255))
        g = int(np.clip(lon_norm * 255, 0, 255))
        b = int(np.clip((row.get('waterfront', 0) * 100 + 100), 0, 255))
        
        # Create 256x256 image with some noise
        img_array = np.zeros((256, 256, 3), dtype=np.uint8)
        img_array[:, :, 0] = r + np.random.randint(-20, 20, (256, 256))
        img_array[:, :, 1] = g + np.random.randint(-20, 20, (256, 256))
        img_array[:, :, 2] = b + np.random.randint(-20, 20, (256, 256))
        img_array = np.clip(img_array, 0, 255)
        
        img = Image.fromarray(img_array)
        img.save(img_path)
    
    print(f"Created placeholder images in {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch satellite images for properties")
    parser.add_argument("--data", type=str, default="data/raw/train.csv",
                        help="Path to CSV with property data")
    parser.add_argument("--output", type=str, default="data/images",
                        help="Output directory for images")
    parser.add_argument("--provider", type=str, default="mapbox",
                        choices=["mapbox", "google", "osm"],
                        help="API provider for satellite images")
    parser.add_argument("--placeholder", action="store_true",
                        help="Create placeholder images instead of fetching")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    if args.placeholder:
        create_placeholder_images(df, args.output)
    else:
        fetcher = SatelliteImageFetcher(api_provider=args.provider, output_dir=args.output)
        fetcher.fetch_all_images(df)
