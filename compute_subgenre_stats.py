#!/usr/bin/env python3
"""
Compute comprehensive statistics for each subgenre in the music dataset.

This script analyzes audio files organized by genre/subgenre structure and computes:
- Track count per subgenre
- Mean duration per subgenre
- BPM/tempo statistics (mean, std, distribution)
- Harmonic-to-percussive energy ratio (HPSS)
- Pitch class profiles (chroma) averaged per subgenre

Results are saved as CSV files and visualizations in the specified output directory.
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
import random
from scipy import stats

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

class SubgenreAnalyzer:
    def __init__(self, wav_dir='WAV', output_dir='subgenre_stats', sr=22050, max_tracks_per_subgenre=35):
        self.wav_dir = Path(wav_dir)
        self.output_dir = Path(output_dir)
        self.sr = sr
        self.max_tracks_per_subgenre = max_tracks_per_subgenre
        self.output_dir.mkdir(exist_ok=True)

        # Initialize results storage
        self.stats = {
            'subgenre': [],
            'genre': [],
            'track_count': [],
            'mean_duration': [],
            'std_duration': [],
            'mean_bpm': [],
            'std_bpm': [],
            'mean_hpss_ratio': [],
            'std_hpss_ratio': []
        }

        # Store chroma profiles separately due to dimensionality
        self.chroma_profiles = {}
        self.tempo_data = {}

    def find_audio_files(self):
        """Find all audio files organized by genre/subgenre."""
        audio_files = {}

        for genre_dir in self.wav_dir.iterdir():
            if not genre_dir.is_dir():
                continue

            genre_name = genre_dir.name

            for subgenre_dir in genre_dir.iterdir():
                if not subgenre_dir.is_dir():
                    continue

                subgenre_name = subgenre_dir.name
                key = f"{genre_name}/{subgenre_name}"

                # Find all audio files in subgenre directory
                files = []
                for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                    files.extend(glob.glob(str(subgenre_dir / ext)))

                if files:
                    audio_files[key] = files

        return audio_files

    def correct_octave_error(self, tempo):
        """
        Correct common octave errors in tempo detection for EDM.
        Based on research showing factors of 2, 1/2, 3, 1/3 are common errors.
        """
        # Define typical EDM tempo ranges by genre
        edm_ranges = [
            (120, 140),  # House, Techno
            (140, 180),  # Hardstyle, Trance
            (80, 120),   # Downtempo, Chill
        ]

        # Check if current tempo is already in a valid range
        for min_bpm, max_bpm in edm_ranges:
            if min_bpm <= tempo <= max_bpm:
                return tempo

        # Try octave corrections
        corrections = [2.0, 0.5, 3.0, 1/3.0, 1.5, 2/3.0]

        for factor in corrections:
            corrected = tempo * factor
            # Check if corrected tempo falls in valid EDM range
            for min_bpm, max_bpm in edm_ranges:
                if min_bpm <= corrected <= max_bpm:
                    return corrected

        # If no correction works, return original tempo
        return tempo

    def estimate_tempo_robust(self, y, track_name=""):
        """
        Simple but effective tempo detection with octave correction.
        Returns (tempo, y_harmonic, y_percussive) to avoid recomputing HPSS.
        """
        # Compute HPSS once for efficiency
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        try:
            print(f"      Percussive audio stats: len={len(y_percussive)}, rms={np.sqrt(np.mean(y_percussive**2)):.3f}")

            # Use simple method on percussive component for better beat detection
            tempo, beats = librosa.beat.beat_track(y=y_percussive, sr=self.sr)
            tempo_val = float(tempo)

            print(f"      Raw librosa result: {tempo_val:.6f} BPM")
            print(f"      Number of beats detected: {len(beats)}")

            # Apply octave correction for EDM
            corrected_tempo = self.correct_octave_error(tempo_val)

            print(f"      Final tempo: {corrected_tempo:.1f} BPM (corrected from {tempo_val:.1f})")
            return corrected_tempo, y_harmonic, y_percussive

        except Exception as e:
            print(f"    Tempo detection FAILED for {track_name}: {e}")
            return None, None, None

    def analyze_track(self, file_path):
        """Analyze a single track and return computed features."""
        try:
            # Load audio skipping intro (first 15 seconds) and limit duration for speed
            skip_intro = 15.0  # seconds
            max_duration = 90.0  # seconds after intro
            y, _ = librosa.load(file_path, sr=self.sr, offset=skip_intro, duration=max_duration)
            duration = len(y) / self.sr

            # DEBUG: Show audio characteristics
            track_name = Path(file_path).name
            print(f"    DEBUG: Loading {track_name}")
            print(f"      Audio length: {len(y)} samples ({duration:.1f}s)")
            print(f"      Audio stats: min={np.min(y):.3f}, max={np.max(y):.3f}, rms={np.sqrt(np.mean(y**2)):.3f}")

            # Improved tempo estimation for dance music (returns HPSS components too)
            tempo, y_harmonic, y_percussive = self.estimate_tempo_robust(y, track_name)
            if tempo is None:
                # Skip this track if tempo estimation failed completely
                return None

            # Use HPSS components from tempo estimation (no recomputation needed)
            harmonic_energy = np.sum(y_harmonic ** 2)
            percussive_energy = np.sum(y_percussive ** 2)
            hpss_ratio = harmonic_energy / (percussive_energy + 1e-8)  # Avoid division by zero

            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
            chroma_mean = np.mean(chroma, axis=1)
            # Normalize to unit sum (probability distribution over pitch classes)
            chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)

            return {
                'duration': duration,
                'tempo': tempo,
                'hpss_ratio': hpss_ratio,
                'chroma': chroma_mean
            }

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def analyze_subgenre(self, subgenre_key, file_list):
        """Analyze all tracks in a subgenre."""
        print(f"Analyzing {subgenre_key}...")

        genre_name, subgenre_name = subgenre_key.split('/')

        # Use first N tracks for equal representation across subgenres
        sampled_files = file_list[:self.max_tracks_per_subgenre]
        print(f"  Using {len(sampled_files)}/{len(file_list)} tracks (max {self.max_tracks_per_subgenre})")

        durations = []
        tempos = []
        hpss_ratios = []
        chroma_profiles = []

        valid_tracks = 0

        for file_path in tqdm(sampled_files, desc=f"Processing {subgenre_name}"):
            result = self.analyze_track(file_path)

            if result is not None:
                durations.append(result['duration'])
                tempos.append(result['tempo'])
                hpss_ratios.append(result['hpss_ratio'])
                chroma_profiles.append(result['chroma'])
                valid_tracks += 1

        if valid_tracks == 0:
            print(f"No valid tracks found for {subgenre_key}")
            return

        # Compute statistics
        self.stats['subgenre'].append(subgenre_name)
        self.stats['genre'].append(genre_name)
        self.stats['track_count'].append(valid_tracks)
        self.stats['mean_duration'].append(np.mean(durations))
        self.stats['std_duration'].append(np.std(durations))
        self.stats['mean_bpm'].append(np.mean(tempos))
        self.stats['std_bpm'].append(np.std(tempos))
        self.stats['mean_hpss_ratio'].append(np.mean(hpss_ratios))
        self.stats['std_hpss_ratio'].append(np.std(hpss_ratios))

        # Store chroma profile and tempo data for visualizations
        self.chroma_profiles[subgenre_key] = np.mean(chroma_profiles, axis=0)
        self.tempo_data[subgenre_key] = tempos

    def create_tempo_distributions(self):
        """Create tempo distribution visualizations."""
        if not self.tempo_data:
            print("No tempo data available for visualization")
            return

        # Create figure with subplots
        n_subgenres = len(self.tempo_data)
        n_cols = 3
        n_rows = (n_subgenres + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.flatten()

        for i, (subgenre_key, tempos) in enumerate(self.tempo_data.items()):
            genre, subgenre = subgenre_key.split('/')

            ax = axes[i]
            ax.hist(tempos, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{subgenre}\n({genre})')
            ax.set_xlabel('BPM')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

            # Add statistics text
            mean_bpm = np.mean(tempos)
            std_bpm = np.std(tempos)
            ax.axvline(mean_bpm, color='red', linestyle='--', alpha=0.8)
            ax.text(0.02, 0.98, f'Mean: {mean_bpm:.1f}\nStd: {std_bpm:.1f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Hide empty subplots
        for i in range(len(self.tempo_data), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'tempo_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create multiple improved comparison visualizations
        self.create_tempo_boxplot()
        self.create_tempo_violin_plot()

    def create_tempo_boxplot(self):
        """Create box plot comparison of tempo distributions."""
        if not self.tempo_data:
            return

        # Prepare data for box plot
        data = []
        labels = []

        for subgenre_key, tempos in self.tempo_data.items():
            genre, subgenre = subgenre_key.split('/')

            # Apply same discrete noise as violin plot for consistency
            tempos_array = np.array(tempos, dtype=float)
            unique_count = len(set(tempos))

            if unique_count <= 3 and len(tempos_array) > 3:
                # Add small discrete integer offsets to enable proper boxplot rendering
                discrete_offsets = [-2, -1, 0, 1, 2]
                noise = np.random.choice(discrete_offsets, len(tempos_array))
                tempos_array = tempos_array + noise
                print(f"Boxplot: Added discrete noise to {subgenre} for visualization")

            data.append(tempos_array.tolist())
            labels.append(f'{subgenre}\n({genre})')

        plt.figure(figsize=(12, 8))

        # Create box plot with custom colors
        box_plot = plt.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes by genre
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.xlabel('Subgenre')
        plt.ylabel('BPM')
        plt.title('Tempo Distribution Comparison (Box Plot)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tempo_distributions_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()

    def create_tempo_violin_plot(self):
        """Create violin plot showing full distribution shapes."""
        if not self.tempo_data:
            return

        # Prepare data in long format for seaborn
        plot_data = []
        for subgenre_key, tempos in self.tempo_data.items():
            genre, subgenre = subgenre_key.split('/')

            # Debug: Check data quality for each subgenre
            print(f"Debug: {subgenre} has {len(tempos)} tempo values")
            print(f"  Range: {min(tempos):.1f} - {max(tempos):.1f}")
            unique_count = len(set(tempos))
            print(f"  Unique values: {unique_count}")

            # Add minimal discrete noise if too few unique values for violin plot
            tempos_array = np.array(tempos, dtype=float)
            if unique_count <= 3 and len(tempos_array) > 3:
                # Add small discrete integer offsets to enable violin plot rendering
                discrete_offsets = [-2, -1, 0, 1, 2]
                noise = np.random.choice(discrete_offsets, len(tempos_array))
                tempos_array = tempos_array + noise
                print(f"  Added discrete noise from {discrete_offsets} for visualization")

            for tempo in tempos_array:
                # Ensure tempo is a valid number
                if pd.isna(tempo) or not np.isfinite(tempo):
                    print(f"  Warning: Invalid tempo {tempo} in {subgenre}")
                    continue

                plot_data.append({
                    'subgenre': subgenre,
                    'genre': genre,
                    'label': f'{subgenre}\n({genre})',
                    'bpm': float(tempo)  # Ensure it's a proper float
                })

        if not plot_data:
            print("Error: No valid data for violin plot")
            return

        df_plot = pd.DataFrame(plot_data)

        # Debug: Check dataframe
        print(f"Debug: DataFrame shape: {df_plot.shape}")
        print("Debug: Data types:")
        print(df_plot.dtypes)

        # Check for any remaining data issues
        for label in df_plot['label'].unique():
            subset = df_plot[df_plot['label'] == label]['bpm']
            print(f"  {label}: {len(subset)} values, std={subset.std():.3f}")

        plt.figure(figsize=(12, 8))

        # Create violin plot with inner quartiles explicitly enabled
        sns.violinplot(data=df_plot, x='label', y='bpm', palette='Set3',
                      inner='quart',  # Force inner quartile lines
                      linewidth=1.2)

        plt.xlabel('Subgenre')
        plt.ylabel('BPM')
        plt.title('Tempo Distribution Shapes (Violin Plot)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tempo_distributions_violin.png', dpi=300, bbox_inches='tight')
        plt.close()


    def create_chroma_heatmap(self):
        """Create chroma profile heatmap."""
        if not self.chroma_profiles:
            print("No chroma data available for visualization")
            return

        # Prepare data for heatmap
        chroma_matrix = []
        labels = []

        for subgenre_key, chroma in self.chroma_profiles.items():
            genre, subgenre = subgenre_key.split('/')
            chroma_matrix.append(chroma)
            labels.append(f'{subgenre}\n({genre})')

        chroma_matrix = np.array(chroma_matrix)

        # Normalize each subgenre's chroma profile to make values more interpretable
        # Convert to relative strength (0-1 scale) where 1 = strongest pitch class for that subgenre
        chroma_matrix_normalized = np.zeros_like(chroma_matrix)
        for i in range(chroma_matrix.shape[0]):
            chroma_matrix_normalized[i] = chroma_matrix[i] / (np.max(chroma_matrix[i]) + 1e-8)

        # Create heatmap
        plt.figure(figsize=(10, 8))

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        sns.heatmap(chroma_matrix_normalized,
                   xticklabels=note_names,
                   yticklabels=labels,
                   cmap='YlOrRd',
                   annot=False,
                   cbar_kws={'label': 'Relative Pitch Class Strength (0-1)'},
                   vmin=0,
                   vmax=1)

        plt.title('Average Pitch Class Profiles by Subgenre')
        plt.xlabel('Pitch Class')
        plt.ylabel('Subgenre')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'chroma_profiles_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self):
        """Save all results to CSV files."""
        # Main statistics
        df_stats = pd.DataFrame(self.stats)
        df_stats.to_csv(self.output_dir / 'subgenre_statistics.csv', index=False)

        # Chroma profiles
        if self.chroma_profiles:
            chroma_data = []
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

            for subgenre_key, chroma in self.chroma_profiles.items():
                genre, subgenre = subgenre_key.split('/')
                row = {'genre': genre, 'subgenre': subgenre}
                for i, note in enumerate(note_names):
                    row[f'chroma_{note}'] = chroma[i]
                chroma_data.append(row)

            df_chroma = pd.DataFrame(chroma_data)
            df_chroma.to_csv(self.output_dir / 'chroma_profiles.csv', index=False)

        # Tempo raw data (for further analysis)
        tempo_rows = []
        for subgenre_key, tempos in self.tempo_data.items():
            genre, subgenre = subgenre_key.split('/')
            for tempo in tempos:
                tempo_rows.append({'genre': genre, 'subgenre': subgenre, 'bpm': tempo})

        if tempo_rows:
            df_tempo = pd.DataFrame(tempo_rows)
            df_tempo.to_csv(self.output_dir / 'tempo_raw_data.csv', index=False)

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("Finding audio files...")
        audio_files = self.find_audio_files()

        if not audio_files:
            print("No audio files found!")
            return

        print(f"Found {len(audio_files)} subgenres to analyze")

        # Analyze each subgenre
        for subgenre_key, file_list in audio_files.items():
            self.analyze_subgenre(subgenre_key, file_list)

        # Create visualizations
        print("Creating tempo distribution plots...")
        self.create_tempo_distributions()

        print("Creating chroma profile heatmap...")
        self.create_chroma_heatmap()

        # Save results
        print("Saving results...")
        self.save_results()

        print(f"Analysis complete! Results saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compute subgenre statistics from audio files')
    parser.add_argument('--output_dir', help='Output directory for statistics and visualizations')
    parser.add_argument('--wav_dir', default='WAV', help='Directory containing genre/subgenre audio files (default: WAV)')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for audio analysis (default: 16000)')
    parser.add_argument('--max_tracks', type=int, default=54, help='Maximum tracks per subgenre to analyze (default: 35)')

    args = parser.parse_args()

    analyzer = SubgenreAnalyzer(
        wav_dir=args.wav_dir,
        output_dir=args.output_dir,
        sr=args.sr,
        max_tracks_per_subgenre=args.max_tracks
    )

    analyzer.run_analysis()


if __name__ == '__main__':
    main()