# Main Goal 
Goal of the project: Perceptual Music Similarity Learning

I am finetuning the Pretrained AST Model with a contrastive learning triplet loss approach so that the model learns to separate subgenres of music. These subgenres are defined simply via a corresponding Playlist (collection of individual tracks) and may only have subtle, nuanced differences, but are clearly seperable for me based on my subjective perception and feel.

# Data
The genres and subgenres I train the network on as of right now are:
Techno:
  - Emotional Techno
  - Dark Techno

House:
  - Chill House
  - Party House

Goa:
  - Chill Goa
  - Party Goa

Harder Styles:
  - Zyzz Hardstyle

All of which are electronic music. Each subgenre currently features between 52 and 90 tracks, stored in .WAV format. The genres and subgenres are mutually exclusive.

# Training
The Main Training script is AST_Triplet_training.py. It works with preprocessed features that have been extracted from 10.24s song chunks with the AST feature extractor, this happens in the script Preprocess_AST_features.py.
The Training is configurable with a JSON config File. Here various Hyperparameters and Flags can be set. The structure of the config File is defined in config.py.
The currently used config file is train_from_precomputed_A40.json.

# Evaluation
The resulting embeddings produced by the fine-tuned model are evaluated quantitatively and qualitatively. Subgenre cluster centroids are computed as well as embeddings for the test set part of the dataset tracks. The centroids and test tracks embeddings are visuialized with UMAP. Also, an accuracy is computed by determining for each test track the cosine similarities to the different cluster centroids and determining the closest subgenre centroid as predicted subgenre and comparing that to the true subgenre that the track belongs to.

# Recommendation Part
Another Goal of this work is to use the fine-tuned model for some type of song recommendations. This could for example Playlist Generation by Nearest Neighbour Search based on a starting track. Or to find the most similar 10 tracks for a given track.