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
    python3 -m venv nlp_task_mia_env
    pip install -r requirements.txt
    
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

   
7. **Current logs are in notebook folder**:
https://github.com/ShaistaShabbir-prog/AIX/blob/dev/NLP-task-MIA%20/notebooks/MIA_TASK.ipynb
---

# Model Training and Evaluation Summary

## **Model Training Results**
- The model training completed successfully in **3 epochs**.  
- **Accuracy Trends**:  
  - The training accuracy steadily improved over the epochs, reaching a final accuracy of **91.76%**.  
  - The validation accuracy peaked at **84.76%** during epoch 1 and then decreased to approximately **81%** by epoch 3.  

- **Loss Trends**:  
  - Both training and validation loss decreased over the epochs, suggesting effective learning by the model.  

## **Final Performance Metrics**
- **Training Accuracy**: ~91.76%  
- **Validation Accuracy**: ~81.90%  

### **Base Model Classification Report**  
| Label | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.8252    | 0.8094 | 0.8172   | 12,500  |
| 1     | 0.8130    | 0.8286 | 0.8207   | 12,500  |
| **Accuracy** | **-**     | **-**    | **0.8190** | **25,000** |

- The classification report indicates a good balance between precision, recall, and F1-score, with an overall accuracy of **81.90%**.  

### **Confusion Matrix**
The confusion matrix shows:  
- Correctly classified **10,117 positive samples** and **10,357 negative samples**.  
- Misclassifications:  
  - **2,383 negative samples** were classified as positive.  
  - **2,143 positive samples** were classified as negative.  

---

## **Membership Inference Attack (MIA) Analysis**

### **Baseline Model Results**
- **MIA Accuracy (Before Privacy Preservation)**: **1.0000**  
- **MIA ROC-AUC**: **1.0000**  

The MIA achieved 100% accuracy on the baseline model, indicating severe overfitting. This overfitting makes it easy for the MIA to distinguish between members and non-members of the training set. The model's high confidence in predictions reveals too much information, undermining privacy.  

---

## **Privacy Preservation Attempts**
To mitigate the overfitting and improve privacy, the following methods were attempted:  
1. **Regularization**  
2. **Early Stopping**

### **Results After Privacy Preservation**  
- **MIA Accuracy (After Privacy Preservation)**: **1.0000**  
- **MIA ROC-AUC**: **1.0000**  

Despite the interventions, MIA performance remained unchanged, suggesting:  
1. The model is still exposing a significant amount of information.  
2. The applied regularization techniques were insufficient to achieve meaningful privacy preservation.  

---

## **Challenges and Future Work**
- **Differential Privacy Implementation**:  
  Efforts to implement differential privacy were hindered by unresolved import errors in the official documentation. Further exploration and debugging are required.  

- **Regularization Impact**:  
  While regularization and early stopping reduced the model’s performance to avoid overfitting, they may have caused underfitting, as seen in the drop in validation accuracy.  

- **Future Directions**:  
  - Experiment with advanced privacy-preserving techniques such as differential privacy, adversarial regularization, or noise injection.  
  - Conduct more detailed experiments to evaluate the trade-offs between utility and privacy.  

Time constraints limited the exploration of additional privacy-preserving approaches, but the results highlight the critical need for robust techniques to address MIA vulnerabilities.  

--- 



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
