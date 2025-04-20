# DS-AGC-Colab

---

[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=theJingqiZhou/DS-AGC-Colab) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/theJingqiZhou/DS-AGC-Colab/blob/ae76ba51757460ae10ba0700e5aa42b1e6be408a/main.ipynb)

A Pytorch re-implementation of paper ["Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-based Emotion Recognition"](https://doi.org/10.1109/TAFFC.2024.3433470) by Ye et al., based on the [official repository](https://github.com/Vesan-yws/DS-AGC.git) and the [Google Colaboratory platform](https://colab.google/).

# Usage

This project involves two main stages: data preprocessing using MATLAB and model training/evaluation using Python, primarily on Google Colaboratory.

## Data Preparation (MATLAB Required)

**Crucial Prerequisite:** Raw EEG data must be preprocessed using MATLAB before running the Python notebook.

1.  **Download Datasets:** Obtain the desired SEED datasets (see [Datasets](#datasets) below). Note: Access often requires credentials.
2.  **Preprocess Data:**
  *   **Run `startup.m`:** Before running any preprocessing script, execute `startup.m` in the project root directory from within MATLAB. This adds the `scripts/` directory to the MATLAB search path and ensures the current working directory remains at the project root, allowing scripts to correctly locate the `data/` directory.
  *   Use the MATLAB scripts provided in the `scripts/` directory (e.g., `seed.m`, `seed_iv.m`) to extract features like Differential Entropy (DE). Adapt scripts if necessary for your specific data format.
  *   Run these scripts using either:
    *   A local MATLAB installation.
    *   MATLAB Online (click the badge at the top; requires a MathWorks account and potentially uploading data to MATLAB Drive).
3.  **Upload Processed Data:** Upload the resulting `.mat` files (containing the extracted features) to your Google Drive or another location accessible by Colab.

## Running the Model (Python/Colab)

1.  **Open in Colab:** Click the "Open In Colab" badge at the top of this README.
2.  **Configure Notebook:**
  *   Mount your Google Drive within the Colab environment (e.g., using `from google.colab import drive; drive.mount('/content/drive')`).
  *   In the [main.ipynb](main.ipynb) notebook, update the file path variables to point to your preprocessed `.mat` files stored on Google Drive.
3.  **Run:** Execute the cells in the notebook sequentially to train and evaluate the model.

*(Alternatively, you can clone the repository, set up a compatible local Python environment, ensure data paths are correct, and run the [main.ipynb](main.ipynb) notebook using Jupyter or a similar tool.)*

# Datasets

This implementation is primarily designed for the SJTU Emotion EEG Datasets:

*   [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html)
*   [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html)
*   [SEED-V](https://bcmi.sjtu.edu.cn/~seed/seed-v.html)

*Note: Access to these datasets typically requires applying for credentials from the providers.*

While the [FACED dataset](https://doi.org/10.7303/syn50614194) could potentially be adapted, the current data loading code ([datasets.py](datasets.py)) does not explicitly support it. FACED may also have geographical or other access restrictions.

# Code Details

*   **Main Pipeline:** [main.ipynb](main.ipynb) - Jupyter Notebook orchestrating data loading, model training, and evaluation using Python/PyTorch.
*   **Data Loading:** `datasets/` directory - Contains Python modules (e.g., `seed.py`, `seed_iv.py`) for loading preprocessed `.mat` feature files for specific datasets.
*   **Model Implementation:** [modules/models.py](modules/models.py) - Python script defining the DS-AGC model architecture.
*   **Loss Functions:** [modules/losses.py](modules/losses.py) - Python script implementing the necessary loss functions.
*   **MATLAB Preprocessing:** `scripts/` directory - Contains MATLAB scripts (e.g., `../startup.m`, `seed.m`, `seed_iv.m`) for preprocessing raw EEG data and extracting features.

# Citations (BibTeX)

## Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-based Emotion Recognition

```Text
@ARTICLE{10609510,
  author={Ye, Weishan and Zhang, Zhiguo and Teng, Fei and Zhang, Min and Wang, Jianhong and Ni, Dong and Li, Fali and Xu, Peng and Liang, Zhen},
  journal={IEEE Transactions on Affective Computing},
  title={Semi-Supervised Dual-Stream Self-Attentive Adversarial Graph Contrastive Learning for Cross-Subject EEG-Based Emotion Recognition},
  year={2025},
  volume={16},
  number={1},
  pages={290-305},
  keywords={Feature extraction;Emotion recognition;Brain modeling;Electroencephalography;Streams;Data models;Transfer learning;EEG;emotion recognition;graph contrastive learning;domain adaption;semi-supervised learning},
  doi={10.1109/TAFFC.2024.3433470}
}
```

## The SJTU Emotion EEG Dataset (SEED)

1. SEED

```Text
@article{zheng2015investigating,
  title={Investigating Critical Frequency Bands and Channels for {EEG}-based Emotion Recognition with Deep Neural Networks},
  author={Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Autonomous Mental Development},
  doi={10.1109/TAMD.2015.2431497},
  year={2015},
  volume={7},
  number={3},
  pages={162-175},
  publisher={IEEE}
}
```

```Text
@inproceedings{duan2013differential,
  title={Differential entropy feature for {EEG}-based emotion classification},
  author={Duan, Ruo-Nan and Zhu, Jia-Yi and Lu, Bao-Liang},
  booktitle={6th International IEEE/EMBS Conference on Neural Engineering (NER)},
  pages={81--84},
  year={2013},
  organization={IEEE}
}
```

2. SEED-IV

```Text
@ARTICLE{8283814,
  author={W. Zheng and W. Liu and Y. Lu and B. Lu and A. Cichocki},
  journal={IEEE Transactions on Cybernetics},
  title={EmotionMeter: A Multimodal Framework for Recognizing Human Emotions},
  year={2018},
  volume={},
  number={},
  pages={1-13},
  keywords={Electroencephalography;Emotion recognition;Electrodes;Feature extraction;Human computer interaction;Biological neural networks;Brain modeling;Affective brain-computer interactions;deep learning;EEG;emotion recognition;eye movements;multimodal deep neural networks},
  doi={10.1109/TCYB.2018.2797176},
  ISSN={2168-2267},
  month={},
}
```

3. SEED-V

```Text
@article{liu2021comparing,
  title={Comparing Recognition Performance and Robustness of Multimodal Deep Learning Models for Multimodal Emotion Recognition},
  author={Liu, Wei and Qiu, Jie-Lin and Zheng, Wei-Long and Lu, Bao-Liang},
  journal={IEEE Transactions on Cognitive and Developmental Systems},
  year={2021},
  publisher={IEEE}
}
```