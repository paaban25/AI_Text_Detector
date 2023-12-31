# AI Text Detector

This project is aimed at detecting whether a given text is human-written or generated by an AI.

## Project Structure

The project is organized into the following directories:

- `dataset`: Contains the dataset Excel file.
- `src`: Source code files for data preprocessing, model training, and prediction.
- `models`: Directory to store trained models.
- `notebooks`: Jupyter Notebooks for exploratory data analysis and other analyses.

## Getting Started

### Prerequisites

- Python 3.x
- Install dependencies: `pip install -r requirements.txt`

### Project Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/paaban25/AI_Text_Detector.git
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

### Model Training

Train the AI Text Detector model:

```bash
python main.py

###Prediction

Predict any Essay whether it is AI generated or Human written

```bash
python src/predict.py

