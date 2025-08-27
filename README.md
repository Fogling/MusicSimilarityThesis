# Main Goal 
Goal of the project: Perceptual Music Similarity Learning

I am finetuning the Pretrained AST Model with a contrastive learning triplet loss approach so that the model learns to separate subgenres of music. These subgenres are defined simply via a corresponding Playlist (collection of individual tracks) and may only have subtle, nuanced differences. 

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
The Main Training script is AST_Triplet_training.py. It works with preprocessed features that have been extracted with the AST feature extractor, this happens in the script Preprocess_AST_features.py.
The Training is configurable with a JSON config File. Here various Hyperparameters and Flags can be set. The structure of the config File is defined in config.py.
The currently used config file is train_from_precomputed.json.

# Computing
I am executing the training of a remote compute cluster with an Nvidia A40 GPU and a lot of RAM and CPU Power. Refer to the SLURM Script Train_from_precomputed_A40.sbatch for more Information.

# Current Challenges
Struggling with Overfitting or Underfitting. Finding the right hyperparameter set for optimal results.
