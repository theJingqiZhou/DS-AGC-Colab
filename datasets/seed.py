import os
from typing import Any

import numpy as np
from scipy import io as sio
from sklearn import preprocessing

# Constants for SEED dataset
_NUM_SUBJECTS = 15
_NUM_CLASSES = 3
_NUM_CHANNELS = 62
_NUM_BANDS = 5


def _get_session_data(
    file_path: str, session_num: int
) -> tuple[np.ndarray, np.ndarray]:
    """Loads feature and label data from a .mat file for a specific session."""
    try:
        data = sio.loadmat(file_path)
        session_key = f"dataset_session{session_num}"
        if session_key not in data:
            raise KeyError(f"Key '{session_key}' not found in MAT file: {file_path}")
        features: np.ndarray = data[session_key]["feature"][0, 0]
        labels: np.ndarray = data[session_key]["label"][0, 0]
        labels = labels.flatten()  # Ensure labels are 1D
        return features.astype(np.float32), labels.astype(int)
    except Exception as e:
        print(f"Error loading data from {file_path} for session {session_num}: {e}")
        raise


def _one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Converts 1D integer labels to one-hot encoded format."""
    if labels.ndim != 1:
        raise ValueError("Input labels must be a 1D array.")
    if np.min(labels) < 0 or np.max(labels) >= num_classes:
        raise ValueError(
            f"Label values must be between 0 and {num_classes - 1}, "
            f"found min={np.min(labels)}, max={np.max(labels)}"
        )
    return np.eye(num_classes)[labels]


def load_dataset(
    base_dir: str, test_id: int, session_idx: int, params: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Loads and prepares the SEED dataset for a specific session and subject.

    Args:
        base_dir: The root directory containing the SEED dataset features.
        test_id: The index (0-based) of the subject to be used as the target domain.
        session_idx: The index (0-based) of the session to load (0, 1, or 2).
        params: A dictionary containing parameters, including:
            'semi' (int): The semi-supervised strategy.
                0: All source data is labeled.
                1: Some source subjects are fully unlabeled ('num_of_U' determines how many).
                2: Each source subject's data is split into labeled (first 'num_labeled_trials')
                   and unlabeled parts.
            'num_of_U' (int, optional): Number of unlabeled source subjects for semi=1.
                                        Defaults to 0.
            'num_labeled_trials' (int, optional): Number of initial trials to use as
                                                  labeled data for semi=2. Defaults to 3.

    Returns:
        A tuple containing three dictionaries:
        - target_set: {'feature': target_features, 'label': target_labels}
        - source_set_labeled: {'feature': labeled_source_features, 'label': labeled_source_labels}
        - source_set_unlabeled: {'feature': unlabeled_source_features, 'label': unlabeled_source_labels}

    Raises:
        FileNotFoundError: If the dataset directory for the session is not found.
        ValueError: If parameters are invalid or data dimensions mismatch.
        KeyError: If expected keys are missing in the .mat files.
    """
    if not 0 <= test_id < _NUM_SUBJECTS:
        raise ValueError(
            f"test_id must be between 0 and {_NUM_SUBJECTS - 1}, got {test_id}"
        )
    if not 0 <= session_idx < 3:
        raise ValueError(f"session_idx must be between 0 and 2, got {session_idx}")

    session_num = session_idx + 1  # Sessions are 1-based in file paths
    session_path = os.path.join(
        base_dir, "de_feature", f"feature_for_net_session{session_num}_LDS_de"
    )
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Dataset directory not found: {session_path}")

    # --- Configuration ---
    semi_setting = params.get("semi", 0)
    num_unlabeled_subjects = params.get("num_of_U", 0) if semi_setting == 1 else 0
    num_labeled_trials = params.get("num_labeled_trials", 3) if semi_setting == 2 else 0

    # --- Data Structures ---
    source_features_labeled: list[np.ndarray] = []
    source_labels_labeled: list[np.ndarray] = []
    source_features_unlabeled: list[np.ndarray] = []
    source_labels_unlabeled: list[np.ndarray] = []
    target_features_list: list[np.ndarray] = []
    target_labels_list: list[np.ndarray] = []

    # --- Preprocessing Tools ---
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # Trial lengths for SEED (used only if semi_setting == 2)
    # Consider loading this from a file if it's standard for SEED
    video_time = np.array(
        [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206]
    )
    if semi_setting == 2 and num_labeled_trials > len(video_time):
        raise ValueError(
            f"num_labeled_trials ({num_labeled_trials}) cannot exceed the number "
            f"of trials ({len(video_time)})"
        )
    split_point = (
        np.cumsum(video_time[:num_labeled_trials])[-1] if semi_setting == 2 else 0
    )

    # --- Determine Unlabeled Source Subjects (for semi=1) ---
    unlabeled_subject_indices = set()
    if semi_setting == 1:
        for i in range(num_unlabeled_subjects):
            # Wrap around if necessary
            unlabeled_idx = (test_id + 1 + i) % _NUM_SUBJECTS
            unlabeled_subject_indices.add(unlabeled_idx)
        # Ensure the test subject is not accidentally included
        unlabeled_subject_indices.discard(test_id)

    # --- Load and Process Data for Each Subject ---
    subject_files = sorted(os.listdir(session_path))
    if len(subject_files) != _NUM_SUBJECTS:
        print(
            f"Warning: Expected {_NUM_SUBJECTS} subject files in {session_path}, "
            f"found {len(subject_files)}. Processing available files."
        )

    for subject_idx, filename in enumerate(subject_files):
        if not filename.endswith(".mat"):
            print(f"Skipping non-MAT file: {filename}")
            continue

        file_path = os.path.join(session_path, filename)
        try:
            features, labels = _get_session_data(file_path, session_num)
            labels_one_hot = _one_hot_encode(labels, _NUM_CLASSES)

            # Scale features (fit_transform separately for each subject)
            features_scaled = min_max_scaler.fit_transform(features)
            # Reshape features: (n_samples, n_channels, n_bands)
            features_reshaped = features_scaled.reshape(
                features.shape[0], _NUM_CHANNELS, _NUM_BANDS, order="F"
            )

            if subject_idx == test_id:
                # --- Target Domain ---
                target_features_list.append(features_reshaped)
                target_labels_list.append(labels_one_hot)
            else:
                # --- Source Domain ---
                if semi_setting == 1:
                    if subject_idx in unlabeled_subject_indices:
                        source_features_unlabeled.append(features_reshaped)
                        source_labels_unlabeled.append(labels_one_hot)
                    else:
                        source_features_labeled.append(features_reshaped)
                        source_labels_labeled.append(labels_one_hot)
                elif semi_setting == 2:
                    if split_point >= features_reshaped.shape[0]:
                        print(
                            f"Warning: split_point {split_point} >= number of samples "
                            f"{features_reshaped.shape[0]} for subject {subject_idx}. "
                            f"All data treated as labeled."
                        )
                        source_features_labeled.append(features_reshaped)
                        source_labels_labeled.append(labels_one_hot)
                    else:
                        source_features_labeled.append(features_reshaped[:split_point])
                        source_labels_labeled.append(labels_one_hot[:split_point])
                        source_features_unlabeled.append(
                            features_reshaped[split_point:]
                        )
                        source_labels_unlabeled.append(labels_one_hot[split_point:])
                else:  # semi_setting == 0 or default
                    source_features_labeled.append(features_reshaped)
                    source_labels_labeled.append(labels_one_hot)

        except (FileNotFoundError, KeyError, ValueError, TypeError) as e:
            print(f"Skipping subject {subject_idx} due to error: {e}")
            continue  # Skip to the next subject if loading/processing fails

    # --- Aggregate Data ---
    # Define shapes for empty arrays
    feature_shape = (
        0,
        _NUM_CHANNELS,
        _NUM_BANDS,
    )
    label_shape = (0, _NUM_CLASSES)

    target_feature = (
        np.vstack(target_features_list)
        if target_features_list
        else np.empty(feature_shape, dtype=np.float32)
    )
    target_label = (
        np.vstack(target_labels_list)
        if target_labels_list
        else np.empty(label_shape, dtype=np.float32)  # Labels are float after one-hot
    )

    source_feature_labeled = (
        np.vstack(source_features_labeled)
        if source_features_labeled
        else np.empty(feature_shape, dtype=np.float32)
    )
    source_label_labeled = (
        np.vstack(source_labels_labeled)
        if source_labels_labeled
        else np.empty(label_shape, dtype=np.float32)
    )

    source_feature_unlabeled = (
        np.vstack(source_features_unlabeled)
        if source_features_unlabeled
        else np.empty(feature_shape, dtype=np.float32)
    )
    source_label_unlabeled = (
        np.vstack(source_labels_unlabeled)
        if source_labels_unlabeled
        else np.empty(label_shape, dtype=np.float32)
    )

    # --- Prepare Output Dictionaries ---
    target_set = {"feature": target_feature, "label": target_label}
    source_set_labeled = {
        "feature": source_feature_labeled,
        "label": source_label_labeled,
    }
    source_set_unlabeled = {
        "feature": source_feature_unlabeled,
        "label": source_label_unlabeled,
    }

    # --- Print Shapes for Verification (Optional) ---
    # print("Target Set Shapes:", target_set["feature"].shape, target_set["label"].shape)
    # print(
    #     "Labeled Source Set Shapes:",
    #     source_set_labeled["feature"].shape,
    #     source_set_labeled["label"].shape,
    # )
    # print(
    #     "Unlabeled Source Set Shapes:",
    #     source_set_unlabeled["feature"].shape,
    #     source_set_unlabeled["label"].shape,
    # )

    return target_set, source_set_labeled, source_set_unlabeled
