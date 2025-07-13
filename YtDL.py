import yt_dlp
import os
import argparse
import logging
import re
import time
from typing import List, Union

class YouTubeMusicDownloader:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.logger = self._setup_logger()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # MP3 download options
        self.ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
            'ignoreerrors': True,
            'no_warnings': False,
            'quiet': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
        }

    def _setup_logger(self):
        """Setup logging configuration"""
        logger = logging.getLogger('YouTubeMusicDownloader')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _sanitize_filename(self, filename):
        """Sanitize filename for safe saving"""
        # Remove invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # Limit filename length
        if len(filename) > 200:
            filename = filename[:200]
        return filename

    def _download_track(self, url: str) -> bool:
        """Download a single track as MP3"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                ydl.download([url])
                return True
                
        except Exception as e:
            self.logger.error(f"Error downloading track: {e}")
            return False

    def download_playlist(self, playlist_url: str) -> List[str]:
        """Download all tracks from a playlist as MP3"""
        downloaded_tracks = []
        
        try:
            # First get playlist info
            with yt_dlp.YoutubeDL({'quiet': True, 'extract_flat': True}) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if 'entries' not in playlist_info:
                    self.logger.error("No tracks found in playlist")
                    return downloaded_tracks
                
                videos = playlist_info['entries']
                playlist_title = playlist_info.get('title', 'Unknown Playlist')
                
                self.logger.info(f"Found {len(videos)} tracks in playlist: {playlist_title}")
                
                # Download each track
                for i, video in enumerate(videos, 1):
                    video_url = f"https://www.youtube.com/watch?v={video['id']}"
                    title = video.get('title', 'Unknown Track')
                    self.logger.info(f"Downloading track {i}/{len(videos)}: {title}")
                    
                    if self._download_track(video_url):
                        downloaded_tracks.append(title)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(1)
                
                self.logger.info(f"Downloaded {len(downloaded_tracks)} tracks to {self.output_dir}")
                return downloaded_tracks
                
        except Exception as e:
            self.logger.error(f"Error downloading playlist: {e}")
            return downloaded_tracks

    def download_single_video(self, video_url: str) -> bool:
        """Download a single video as MP3"""
        try:
            # Get video info
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                video_info = ydl.extract_info(video_url, download=False)
                title = video_info.get('title', 'Unknown Video')
                
            self.logger.info(f"Downloading track: {title}")
            result = self._download_track(video_url)
            
            if result:
                self.logger.info(f"Successfully downloaded: {title}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error downloading video: {e}")
            return False

    def download(self, url: str) -> Union[List[str], bool]:
        """Detect URL type and download accordingly"""
        try:
            # Check if it's a playlist or a single video
            if 'playlist' in url or 'list=' in url:
                self.logger.info("Detected playlist URL")
                return self.download_playlist(url)
            else:
                self.logger.info("Detected single video URL")
                return self.download_single_video(url)
                
        except Exception as e:
            self.logger.error(f"Error processing URL: {e}")
            return False


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Download YouTube videos/playlists as MP3')
    parser.add_argument('url', help='YouTube video or playlist URL')
    parser.add_argument('--output', '-o', default='music_downloads', 
                        help='Output directory for downloaded MP3s (default: music_downloads)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and use downloader
    downloader = YouTubeMusicDownloader(output_dir=args.output)
    downloader.download(args.url)
    
    print(f"\nDownload completed! Your music is in the '{args.output}' folder.")


if __name__ == "__main__":
    main()