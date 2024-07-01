# Handwriting Digit Recognition

This project uses a neural network to recognize handwritten digits using the MNIST dataset.

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Contains Jupyter Notebooks for experimentation.
- `models/`: Contains saved models.
- `src/`: Contains Python scripts for data processing, model building, training, evaluating, and predicting.
- `reports/`: Contains reports and figures.
- `requirements.txt`: Lists project dependencies.
- `README.md`: Project overview and instructions.

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/handwriting_digit_recognition.git
    cd handwriting_digit_recognition
    ```

2. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Run the Jupyter Notebook:

    ```bash
    jupyter notebook notebooks/digit_recognition.ipynb
    ```

## Usage

- To preprocess data, run:

    ```bash
    python src/data_preprocessing.py
    ```

- To train the model, run:

    ```bash
    python src/train.py
    ```

- To evaluate the model, run:

    ```bash
    python src/evaluate.py
    ```

- To make predictions, run:

    ```bash
    python src/predict.py
    ```
