#!/usr/bin/env python3
"""
Demo script for the Music Recommendation System.

This script demonstrates various features of the recommendation system:
1. Loading a trained model
2. Computing embeddings for tracks
3. Getting single track recommendations
4. Generating playlists from seed tracks
5. Evaluating recommendation quality
"""

import argparse
import logging
from pathlib import Path
import random
from typing import List, Dict

from music_recommender import MusicRecommender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_single_recommendations(recommender: MusicRecommender, num_examples: int = 3):
    """Demonstrate single track recommendations"""
    print("\n" + "="*60)
    print("SINGLE TRACK RECOMMENDATIONS DEMO")
    print("="*60)

    # Get some random tracks to demo
    track_ids = list(recommender.track_embeddings.keys())
    demo_tracks = random.sample(track_ids, min(num_examples, len(track_ids)))

    for track_id in demo_tracks:
        metadata = recommender.track_metadata[track_id]
        print(f"\n🎵 Query Track: {track_id}")
        print(f"   Subgenre: {metadata['subgenre']}")
        print(f"   Chunks used: {metadata['chunks_used']}")

        # Get recommendations
        recommendations = recommender.get_recommendations(track_id, k=5, exclude_same_track=True)

        print("   Top 5 Similar Tracks:")
        for i, (rec_track_id, similarity) in enumerate(recommendations, 1):
            rec_metadata = recommender.track_metadata[rec_track_id]
            same_subgenre = "✓" if rec_metadata['subgenre'] == metadata['subgenre'] else "✗"
            print(f"     {i}. {rec_track_id} (sim: {similarity:.3f}) [{rec_metadata['subgenre']}] {same_subgenre}")


def demo_playlist_generation(recommender: MusicRecommender, num_playlists: int = 2):
    """Demonstrate playlist generation"""
    print("\n" + "="*60)
    print("PLAYLIST GENERATION DEMO")
    print("="*60)

    track_ids = list(recommender.track_embeddings.keys())

    for i in range(num_playlists):
        print(f"\n🎶 Playlist #{i+1}")

        # Select random seed tracks from different subgenres
        subgenres = list(recommender.subgenre_to_tracks.keys())
        num_seeds = min(2, len(subgenres))
        selected_subgenres = random.sample(subgenres, num_seeds)

        seed_tracks = []
        for subgenre in selected_subgenres:
            tracks_in_subgenre = recommender.subgenre_to_tracks[subgenre]
            seed_tracks.append(random.choice(tracks_in_subgenre))

        print(f"Seed tracks: {seed_tracks}")

        # Generate playlist
        playlist = recommender.generate_playlist(
            seed_tracks=seed_tracks,
            playlist_length=10,
            diversity_weight=0.1
        )

        print("Generated Playlist:")
        for j, track_id in enumerate(playlist, 1):
            metadata = recommender.track_metadata[track_id]
            is_seed = "🌱" if track_id in seed_tracks else "  "
            print(f"  {is_seed} {j:2d}. {metadata['subgenre']} - {metadata['track_name']}")


def demo_cross_subgenre_analysis(recommender: MusicRecommender):
    """Analyze cross-subgenre recommendations"""
    print("\n" + "="*60)
    print("CROSS-SUBGENRE ANALYSIS")
    print("="*60)

    subgenre_stats = {}

    for subgenre, tracks in recommender.subgenre_to_tracks.items():
        if len(tracks) < 2:  # Skip subgenres with too few tracks
            continue

        sample_tracks = random.sample(tracks, min(3, len(tracks)))
        cross_genre_counts = {sg: 0 for sg in recommender.subgenre_to_tracks.keys()}
        total_recommendations = 0

        for track_id in sample_tracks:
            recommendations = recommender.get_recommendations(
                track_id, k=5, exclude_same_track=True
            )

            for rec_track_id, _ in recommendations:
                rec_subgenre = recommender.track_metadata[rec_track_id]['subgenre']
                cross_genre_counts[rec_subgenre] += 1
                total_recommendations += 1

        if total_recommendations > 0:
            subgenre_stats[subgenre] = {
                'same_genre_ratio': cross_genre_counts[subgenre] / total_recommendations,
                'most_similar_other_genre': max(
                    [(sg, count) for sg, count in cross_genre_counts.items() if sg != subgenre],
                    key=lambda x: x[1], default=("None", 0)
                )
            }

    print("Subgenre Similarity Analysis:")
    print("(Shows how often tracks recommend within same subgenre vs. cross-genre)")
    print()

    for subgenre, stats in subgenre_stats.items():
        same_ratio = stats['same_genre_ratio']
        other_genre, other_count = stats['most_similar_other_genre']
        print(f"{subgenre}:")
        print(f"  Same-genre ratio: {same_ratio:.2f}")
        print(f"  Most similar other genre: {other_genre} ({other_count} recommendations)")
        print()


def evaluate_recommendation_quality(recommender: MusicRecommender, num_queries: int = 20):
    """Evaluate recommendation quality using simple metrics"""
    print("\n" + "="*60)
    print("RECOMMENDATION QUALITY EVALUATION")
    print("="*60)

    track_ids = list(recommender.track_embeddings.keys())
    test_tracks = random.sample(track_ids, min(num_queries, len(track_ids)))

    same_subgenre_hits = 0
    total_recommendations = 0
    similarity_scores = []

    for track_id in test_tracks:
        query_subgenre = recommender.track_metadata[track_id]['subgenre']
        recommendations = recommender.get_recommendations(track_id, k=5, exclude_same_track=True)

        for rec_track_id, similarity in recommendations:
            rec_subgenre = recommender.track_metadata[rec_track_id]['subgenre']

            if rec_subgenre == query_subgenre:
                same_subgenre_hits += 1

            similarity_scores.append(similarity)
            total_recommendations += 1

    # Calculate metrics
    same_subgenre_precision = same_subgenre_hits / total_recommendations if total_recommendations > 0 else 0
    avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
    min_similarity = min(similarity_scores) if similarity_scores else 0
    max_similarity = max(similarity_scores) if similarity_scores else 0

    print(f"Evaluation Results (based on {num_queries} queries):")
    print(f"  Same-subgenre precision: {same_subgenre_precision:.3f}")
    print(f"  Average similarity score: {avg_similarity:.3f}")
    print(f"  Similarity score range: {min_similarity:.3f} - {max_similarity:.3f}")
    print(f"  Total recommendations analyzed: {total_recommendations}")

    # Subgenre distribution in recommendations
    print(f"\nSubgenre Distribution in Recommendations:")
    subgenre_counts = {}
    for track_id in test_tracks:
        recommendations = recommender.get_recommendations(track_id, k=5, exclude_same_track=True)
        for rec_track_id, _ in recommendations:
            rec_subgenre = recommender.track_metadata[rec_track_id]['subgenre']
            subgenre_counts[rec_subgenre] = subgenre_counts.get(rec_subgenre, 0) + 1

    total_recs = sum(subgenre_counts.values())
    for subgenre, count in sorted(subgenre_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_recs) * 100 if total_recs > 0 else 0
        print(f"  {subgenre}: {count} ({percentage:.1f}%)")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="Demo for Music Recommendation System")

    parser.add_argument("--model_dir", type=str, required=True,
                       help="Path to directory containing trained model")
    parser.add_argument("--splits_file", type=str, default=None,
                       help="Path to splits.json file")
    parser.add_argument("--chunks_dir", type=str, default=None,
                       help="Path to chunks directory")
    parser.add_argument("--splits_type", type=str, default="test", choices=["train", "test", "both"],
                       help="Which splits to use")
    parser.add_argument("--load_embeddings", type=str, default=None,
                       help="Load precomputed embeddings from file")
    parser.add_argument("--save_embeddings", type=str, default=None,
                       help="Save computed embeddings to file")
    parser.add_argument("--demo_only", action="store_true",
                       help="Run only demos, skip evaluation")

    args = parser.parse_args()

    print("🎵 Music Recommendation System Demo")
    print("="*60)

    # Initialize recommender
    print("Loading model and computing embeddings...")
    recommender = MusicRecommender(args.model_dir)

    # Load embeddings
    if args.load_embeddings:
        recommender.load_embeddings(args.load_embeddings)
    else:
        recommender.load_embeddings_from_splits(
            splits_file=args.splits_file,
            chunks_dir=args.chunks_dir,
            splits_type=args.splits_type
        )

    # Save embeddings if requested
    if args.save_embeddings:
        recommender.save_embeddings(args.save_embeddings)

    # Print basic statistics
    stats = recommender.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Total subgenres: {stats['total_subgenres']}")
    print(f"  Embedding dimension: {stats['embedding_dimension']}")
    print(f"  Tracks per subgenre: {dict(list(stats['tracks_per_subgenre'].items())[:5])}{'...' if len(stats['tracks_per_subgenre']) > 5 else ''}")

    # Run demos
    demo_single_recommendations(recommender)
    demo_playlist_generation(recommender)
    demo_cross_subgenre_analysis(recommender)

    # Run evaluation if requested
    if not args.demo_only:
        evaluate_recommendation_quality(recommender)

    print("\n" + "="*60)
    print("Demo completed! 🎉")

    # Show example usage
    print("\nExample Usage:")
    print("  # Get recommendations for a specific track")
    print(f"  python music_recommender.py --model_dir {args.model_dir} --query_track 'Subgenre/Track_Name'")
    print("  # Generate playlist from seed tracks")
    print(f"  python music_recommender.py --model_dir {args.model_dir} --generate_playlist --seed_tracks 'Genre1/Track1' 'Genre2/Track2'")


if __name__ == "__main__":
    main()