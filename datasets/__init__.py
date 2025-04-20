"""
Dataset loading for EEG emotion recognition.
Contains functions to load SEED, SEED-IV, and SEED-V datasets.
Organizes data into labeled source, unlabeled source, and target sets.

The dataset loading logic by @author: voicebeer is extracted into this
standalone module to keep the main pipeline code clean.
"""

from datasets import seed, seed_iv, seed_v
