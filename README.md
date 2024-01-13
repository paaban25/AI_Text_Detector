# AI Text Detector

Detect whether a given text is human-written or generated by an AI.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Project Setup](#project-setup)
- [Usage](#usage)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Model Training](#model-training)
  - [Prediction](#prediction)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The AI Text Detector project aims to classify text as either human-written or generated by an AI. The project includes data preprocessing, model training, and prediction components.

## Project Structure

The project is organized into the following directories:

- **`dataset`**: Contains the dataset Excel file (`dataset.xlsx`). The dataset includes two columns: "text" (paragraphs) and "generated" (labels indicating whether the text is human-written (0) or AI-generated (1)).

- **`src`**: Source code files for different stages of the project.
  - **`data_preprocessing.py`**: Code for loading, preprocessing, and splitting the dataset.
  - **`train_model.py`**: Code for training the AI Text Detector model.
  - **`predict.py`**: Code for making predictions using the trained model.

- **`models`**: Directory to store trained models. After running the model training code, the trained model and vectorizer are saved here.

- **`notebooks`**: Jupyter Notebooks for exploratory data analysis and additional analyses. The main notebook is `exploratory_data_analysis.ipynb`.

## Getting Started

### Prerequisites

- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

### Project Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/AI_Text_Detector.git
    cd AI_Text_Detector
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Place your dataset Excel file (`dataset.xlsx`) inside the `dataset` directory.

## Usage

### Exploratory Data Analysis

Explore and analyze the dataset using Jupyter Notebooks:

```bash
cd notebooks
jupyter notebook exploratory_data_analysis.ipynb

### Predicting about the text

Run the predict.py inside the src directory to predict if a given paragraph is human written or AI generated.


