#!/usr/bin/env python3
"""
Create balanced train-test splits for music genre classification.

This script creates a clean, balanced 95-5 train-test split from preprocessed features,
ensuring equal representation of tracks from each subgenre (limited by the smallest subgenre).
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainTestSplitter:
    """Creates balanced train-test splits for audio datasets."""
    
    def __init__(self, features_dir: str, train_ratio: float = 0.95):
        """
        Initialize the splitter.
        
        Args:
            features_dir: Path to preprocessed features directory
            train_ratio: Fraction of data for training (default: 0.95 for 95-5 split)
        """
        self.features_dir = Path(features_dir)
        self.train_ratio = train_ratio
        self.test_ratio = 1.0 - train_ratio
        
        if not self.features_dir.exists():
            raise ValueError(f"Features directory does not exist: {features_dir}")
    
    def count_original_tracks(self, wav_dir: str) -> Dict[str, int]:
        """
        Count original WAV files per subgenre to determine available tracks.
        
        Args:
            wav_dir: Path to original WAV files directory
            
        Returns:
            Dictionary mapping subgenre -> track count
        """
        wav_path = Path(wav_dir)
        track_counts = {}
        
        if not wav_path.exists():
            logger.warning(f"WAV directory not found: {wav_dir}. Using preprocessed features only.")
            return self._count_from_features()
        
        # Navigate the nested structure: WAV/genre/subgenre/*.wav
        for genre_dir in wav_path.iterdir():
            if not genre_dir.is_dir():
                continue
                
            for subgenre_dir in genre_dir.iterdir():
                if not subgenre_dir.is_dir():
                    continue
                    
                # Count WAV files
                wav_files = list(subgenre_dir.glob("*.wav")) + list(subgenre_dir.glob("*.mp3"))
                if wav_files:
                    track_counts[subgenre_dir.name] = len(wav_files)
                    logger.info(f"Subgenre '{subgenre_dir.name}': {len(wav_files)} tracks")
        
        return track_counts
    
    def _count_from_features(self) -> Dict[str, int]:
        """Fallback: count tracks from feature chunks."""
        track_counts = {}
        
        for subgenre_dir in self.features_dir.iterdir():
            if not subgenre_dir.is_dir():
                continue
                
            # Count unique track names by removing chunk suffixes
            chunk_files = list(subgenre_dir.glob("*.pt"))
            track_names = set()
            
            for chunk_file in chunk_files:
                # Remove _chunkN.pt suffix to get original track name
                track_name = chunk_file.stem.rsplit("_chunk", 1)[0]
                track_names.add(track_name)
            
            if track_names:
                track_counts[subgenre_dir.name] = len(track_names)
                logger.info(f"Subgenre '{subgenre_dir.name}': {len(track_names)} tracks (from features)")
        
        return track_counts
    
    def get_track_chunks(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Organize chunk files by subgenre and track.
        
        Returns:
            Dict[subgenre][track_name] = [list of chunk paths]
        """
        subgenre_tracks = defaultdict(lambda: defaultdict(list))
        
        for subgenre_dir in self.features_dir.iterdir():
            if not subgenre_dir.is_dir():
                continue
                
            subgenre = subgenre_dir.name
            chunk_files = sorted(subgenre_dir.glob("*.pt"))
            
            for chunk_file in chunk_files:
                # Extract track name by removing chunk suffix
                track_name = chunk_file.stem.rsplit("_chunk", 1)[0]
                subgenre_tracks[subgenre][track_name].append(str(chunk_file))
        
        return subgenre_tracks
    
    def create_balanced_split(self, wav_dir: str = "WAV") -> Tuple[Dict, Dict]:
        """
        Create balanced train-test split.
        
        Args:
            wav_dir: Path to original WAV files for track counting
            
        Returns:
            Tuple of (train_split_dict, test_split_dict)
        """
        logger.info("Starting balanced train-test split creation...")
        
        # Count original tracks per subgenre
        track_counts = self.count_original_tracks(wav_dir)
        
        if not track_counts:
            raise ValueError("No tracks found in any subgenre")
        
        # Find minimum track count to ensure balance
        min_tracks = min(track_counts.values())
        logger.info(f"Minimum tracks per subgenre: {min_tracks}")
        logger.info(f"Will use {min_tracks} tracks per subgenre for balance")
        
        # Calculate train/test counts
        test_tracks_per_genre = max(1, int(min_tracks * self.test_ratio))
        train_tracks_per_genre = min_tracks - test_tracks_per_genre
        
        logger.info(f"Per subgenre: {train_tracks_per_genre} train, {test_tracks_per_genre} test tracks")
        
        # Get track chunks organized by subgenre and track
        subgenre_tracks = self.get_track_chunks()
        
        train_split = {"tracks": defaultdict(list), "metadata": {}}
        test_split = {"tracks": defaultdict(list), "metadata": {}}
        
        total_train_tracks = 0
        total_test_tracks = 0
        
        # Process each subgenre
        for subgenre, tracks_dict in subgenre_tracks.items():
            if subgenre not in track_counts:
                logger.warning(f"Subgenre {subgenre} not found in WAV directory, skipping")
                continue
                
            available_tracks = list(tracks_dict.keys())
            
            if len(available_tracks) < min_tracks:
                logger.warning(f"Subgenre {subgenre} has only {len(available_tracks)} tracks, "
                             f"expected {min_tracks}. Using all available.")
                tracks_to_use = available_tracks
            else:
                tracks_to_use = available_tracks[:min_tracks]
            
            # Split tracks for this subgenre
            test_tracks = tracks_to_use[:test_tracks_per_genre]
            train_tracks = tracks_to_use[test_tracks_per_genre:]
            
            # Add to splits
            for track_name in train_tracks:
                chunk_paths = tracks_dict[track_name]
                train_split["tracks"][subgenre].append({
                    "track_name": track_name,
                    "chunks": chunk_paths,
                    "num_chunks": len(chunk_paths)
                })
                total_train_tracks += 1
            
            for track_name in test_tracks:
                chunk_paths = tracks_dict[track_name]
                test_split["tracks"][subgenre].append({
                    "track_name": track_name,
                    "chunks": chunk_paths,
                    "num_chunks": len(chunk_paths)
                })
                total_test_tracks += 1
            
            logger.info(f"Subgenre '{subgenre}': {len(train_tracks)} train, {len(test_tracks)} test tracks")
        
        # Add metadata
        metadata = {
            "total_subgenres": len(subgenre_tracks),
            "tracks_per_subgenre": min_tracks,
            "train_tracks_per_subgenre": train_tracks_per_genre,
            "test_tracks_per_subgenre": test_tracks_per_genre,
            "total_train_tracks": total_train_tracks,
            "total_test_tracks": total_test_tracks,
            "train_ratio": self.train_ratio,
            "test_ratio": self.test_ratio,
            "features_dir": str(self.features_dir),
            "subgenres": list(subgenre_tracks.keys())
        }
        
        train_split["metadata"] = metadata
        test_split["metadata"] = metadata
        
        logger.info(f"Split complete: {total_train_tracks} train tracks, {total_test_tracks} test tracks")
        return train_split, test_split
    
    def save_splits(self, train_split: Dict, test_split: Dict, output_dir: str = "splits"):
        """Save train and test splits to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        train_file = output_path / "train_split.json"
        test_file = output_path / "test_split.json"
        
        with open(train_file, 'w') as f:
            json.dump(train_split, f, indent=2)
        
        with open(test_file, 'w') as f:
            json.dump(test_split, f, indent=2)
        
        logger.info(f"Splits saved to {train_file} and {test_file}")
        
        # Also save a summary
        summary_file = output_path / "split_summary.txt"
        with open(summary_file, 'w') as f:
            metadata = train_split["metadata"]
            f.write("Train-Test Split Summary\n")
            f.write("=======================\n\n")
            f.write(f"Total subgenres: {metadata['total_subgenres']}\n")
            f.write(f"Tracks per subgenre: {metadata['tracks_per_subgenre']}\n")
            f.write(f"Train ratio: {metadata['train_ratio']:.1%}\n")
            f.write(f"Test ratio: {metadata['test_ratio']:.1%}\n\n")
            f.write(f"Train tracks per subgenre: {metadata['train_tracks_per_subgenre']}\n")
            f.write(f"Test tracks per subgenre: {metadata['test_tracks_per_subgenre']}\n")
            f.write(f"Total train tracks: {metadata['total_train_tracks']}\n")
            f.write(f"Total test tracks: {metadata['total_test_tracks']}\n\n")
            f.write("Subgenres included:\n")
            for subgenre in sorted(metadata['subgenres']):
                f.write(f"  - {subgenre}\n")
        
        logger.info(f"Summary saved to {summary_file}")

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Create balanced train-test splits for music genre classification"
    )
    parser.add_argument(
        "--features-dir",
        default="preprocessed_features",
        help="Path to preprocessed features directory (default: preprocessed_features)"
    )
    parser.add_argument(
        "--wav-dir",
        default="WAV",
        help="Path to original WAV files directory for track counting (default: WAV)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.95,
        help="Fraction of data for training (default: 0.95)"
    )
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="Output directory for split files (default: splits)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create splitter
        splitter = TrainTestSplitter(args.features_dir, args.train_ratio)
        
        # Create balanced split
        train_split, test_split = splitter.create_balanced_split(args.wav_dir)
        
        # Save splits
        splitter.save_splits(train_split, test_split, args.output_dir)
        
        logger.info("Train-test split creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error creating train-test split: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())