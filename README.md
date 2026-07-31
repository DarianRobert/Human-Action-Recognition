# Human Action Recognition via Advanced Sequence Modelling

This repository contains the code and resources for a Machine Learning sequence modelling project, developed as an investigation into Video Action Recognition (VAR). The primary goal is to allow computational systems to automatically detect, label, and analyse human activities directly from raw, sequential video streams. 

For comprehensive details on the dataset splitting, experimental rigour, and architectural comparisons, please refer to the **COMP703 Project.pdf** document.

### Project Team
*   **Arjun Ramphal**
*   **Darian Robert**
*   **Lokadi Naicker**
*   **Keolin Naicker**

---

## Project Overview

Real-world human behaviour is dynamic, and while deep learning achieves high accuracy in static image classification, sequence-based architectures are required to extract deep temporal patterns. This project transitions from spatial feature extraction to joint spatiotemporal modelling.

Automating action recognition has practical significance in several real-world domains:
*   Smart Healthcare & Eldercare
*   Surveillance & Public Safety
*   Human-Computer Interaction (HCI)

---

## Dataset

The project utilizes the **KTH Action Recognition dataset**. 
*   The dataset categorises human actions into six distinct classes: boxing, handclapping, handwaving, jogging, running, and walking.
*   Data is strictly partitioned into training, validation, and testing sets based on subject IDs to prevent data leakage.
*   The dataset splitting and imbalance profiling logic are locked and saved to a static JSON file for experimental reproducibility.

---

## Methodology

The machine learning pipeline is built entirely in Python, utilizing PyTorch and Optuna for rigorous hyperparameter optimization.

*   **Spatial Feature Extraction:** OpenCV is used to sample exactly 30 frames evenly across each video clip. 
*   **Dimensionality Reduction:** A pre-trained MobileNetV2 architecture acts as the spatial backbone to extract dense 1280-dimensional feature embedding vectors.
*   **Foundational Recurrent Architectures Evaluated:** Standard RNN, GRU, LSTM, and Bidirectional LSTM (BiLSTM).
*   **Advanced Contextual Architectures Evaluated:** Vision Sequence Transformer and Bidirectional State Space Model (SSM / Mamba).

---

## Environment and Setup

This project is optimized and designed to be executed in **Google Colab** with a mounted Google Drive environment. 

Before running the pipeline, you must establish the correct folder architecture in your Google Drive so the scripts can locate the raw data. Set up your Google Drive exactly as follows:

```text
My Drive/
└── comp703_project/
    └── raw_kTH/
        ├── boxing/        (contains .avi clips)
        ├── handclapping/  (contains .avi clips)
        ├── handwaving/    (contains .avi clips)
        ├── jogging/       (contains .avi clips)
        ├── running/       (contains .avi clips)
        └── walking/       (contains .avi clips)
```

The code will automatically mount your Google Drive to `/content/drive` and target the base directory at `/content/drive/MyDrive/comp703_project`. As you execute the scripts, the pipeline will automatically generate the necessary output subfolders (such as `train_test_splits`, `processed_features`, `models`, and `reports`) within this base directory.

### Prerequisites
Ensure you have the following Python libraries installed in your runtime:
`torch`, `torchvision`, `opencv-python`, `optuna`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tqdm`.

---

## Execution Steps

The project is structured into modular code blocks designed to be executed sequentially. 

1.  **Dataset Profiling & Splitting:** Run the first code block to scan the videos, verify subject alignments, and generate a locked `kth_official_splits.json` file to guarantee experimental reproducibility.
2.  **Spatial Feature Extraction:** Run the second code block to sample frames and process them through the MobileNetV2 backbone. This converts the videos into 1280-dimensional sequence matrices saved as `.npy` files.
3.  **Foundational Sequence Modelling:** Run the third code block to execute an isolated Optuna optimization search and deep-train the foundational recurrent architectures (RNN, GRU, LSTM, BiLSTM).
4.  **Advanced Sequence Modelling:** Run the fourth code block to optimize and train the Vision Sequence Transformer and Bidirectional SSM (Mamba) models.
5.  **Benchmarking & Evaluation:** Run the final code block to evaluate all trained models against the validation set. This compiles the diagnostic performance metrics and generates a 6-panel normalized confusion matrix grid.