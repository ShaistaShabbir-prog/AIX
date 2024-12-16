# NLP Membership Inference Attack (MIA) and Privacy Preservation

This project demonstrates the process of performing a Membership Inference Attack (MIA) on a deep learning model trained on the IMDB dataset. The workflow includes:
1. Training an LSTM model on the IMDB dataset.
2. Performing MIA on the trained model.
3. Applying privacy-preserving techniques.
4. Repeating MIA after applying privacy techniques to evaluate the effectiveness of the privacy measures.

## Prerequisites

Ensure you have the following tools and libraries:

- [Python 3.10](https://www.python.org/downloads/)

### Required Packages

To run the project, we have a `requirement.txt` file that contains all the necessary packages for the environment. You can create the environment using this requirements file.

### Steps to Run the Project

1. **Clone the repository** (or copy the files):

    ```bash
         https://github.com/ShaistaShabbir-prog/AIX.git
    ```

2. **Create a virtual environment**:

    To install all required dependencies, use the provided `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    python3 -m venv nlp_task_mia_env
    ```

3. **Activate the environment**:

    After the environment is created, activate it using:

    ```bash
    
    source nlp_task_mia_env/bin/activate
    ```

4. **Run the Project**:

    The main entry point of the project is the `main.py` file. You can run it using:

    ```bash
    python src/main.py
    ```

    This will:
    - Load the IMDB dataset https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data.
    - Train an LSTM model on the dataset.
    - Perform a Membership Inference Attack (MIA)by training a shadow model.
    - Apply privacy-preserving techniques (e.g., L2-regularization, early stopping, adversarial_training, cross validation etc).
    - Re-run MIA and compare the results to assess the effectiveness of the privacy measures.

5. **Monitor Training and Results**:

    The LSTM model will output progress during training, and the Membership Inference Attack results will be displayed in the terminal and saved to log files, you can view these during runtime.

    You can analyze the accuracy, loss, and the impact of privacy techniques by reviewing the results saved during the MIA comparisons.


### File Structure

The project follows this directory structure:
```plaintext
NLP_task_MIA/
│
├── data/
│
├── src/
│   ├── __init__.py            
│   ├── main.py                 # Entry point of the script to execute the LSTM model, MIA, and privacy techniques
│   ├── data_loader.py          # Script for loading and preprocessing the IMDB dataset
│   ├── lstm_model.py           # Script for defining and training the LSTM model
│   ├── attack_model.py         # Script for defining and performing Membership Inference Attack (MIA)
│   └── privacy_preservation.py # Script for applying privacy-preserving techniques (e.g., Differential Privacy)
│   └── model_evaluator.py # different evaluations with visualizations 
│   ├── logs/
│    │   ├── training_logs.txt       # Logs for training LSTM model
│    │   ├── attack_logs.txt         # Logs for Membership Inference Attack
│    │   └── results_comparison.txt  # Logs comparing results before and after privacy-preservation
│
├── notebooks/
│   └──MIA_TASK.ipynb  # Jupyter notebook for experiments and data analysis
│
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project description and instructions
```
