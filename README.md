# Buzzy the Chatbot

## Description
Simple chatbot implemented using TensorFlow and (NLP). The chatbot is trained on a predefined data and can respond to inputs accordingly.

## File Structure

- `preparation.py`: => load, preprocess data, and save data.
- `training.py`: => build and train the model.
- `util.py`: => processing sentences and making predictions.
- `app.py`: => execute the training and test the chatbot.

## Requirements

- Python 3.x
- TensorFlow
- NLTK
- NumPy
- Pickle

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/KhaledAlabssi/Buzzy_the_Chatbot.git
    cd Buzzy_the_Chatbot
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare the intents file (`intents.json`) with the training data "data examples" are shared in (`internts.example.json`).

2. Run the main script to train the model:
    ```bash
    python app.py
    ```

3. The trained model will be saved as `chatbot_model.h5`, and the processed data will be saved as `words.pkl` and `classes.pkl`.

## Get in Touch

> LinkedIn: [linkedIn.com/in/Khaled-Alabssi](https://www.linkedin.com/in/khaled-alabssi/)

> GitHub: [github.com/KhaledAlabssi](https://github.com/KhaledAlabssi)

> Website: [khaled.alabssi.com](https://www.khaled.alabssi.com)

## Happy Coding!