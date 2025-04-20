"""
Module for Dual-Space-Augmented Graph Contrastive Learning.

This package provides implementations for graph contrastive learning
with dual space augmentation. It includes loss functions and models
necessary for the DS-AGC method.

Components:
- AutomaticWeightedLoss: Loss function with automatic weight balancing
- SemiGCL: Semi-supervised Graph Contrastive Learning model

@author: Vesan-yws
@author: mindlab

Note: This module organizes the original standalone Python scripts
into a structured package format.
"""

from modules.losses import AutomaticWeightedLoss
from modules.models import SemiGCL
