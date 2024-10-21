
# Text Classification Using CNN with TensorFlow

This project focuses on classifying text into categories using a Convolutional Neural Network (CNN) implemented in TensorFlow. The dataset used is a Persian text corpus, which undergoes preprocessing steps such as tokenization, stemming, lemmatization, and normalization. The model is trained using word embeddings and evaluated for its performance in classifying texts.

## Requirements

Install the necessary libraries by running the following commands:

```bash
pip install hazm
pip install nltk
pip install tensorflow
pip install pandas
pip install matplotlib
pip install sklearn
pip install mlxtend
```

## Data Preprocessing

1. **Text Normalization:** The Persian texts are normalized using the `hazm` library.
2. **Stemming and Lemmatization:** The texts are further processed by reducing words to their root forms.
3. **Tokenization:** The text is tokenized, converting each document into a sequence of words.
4. **One-Hot Encoding:** Labels are one-hot encoded for training.

## Model Architecture

The model is a Sequential CNN with the following layers:

- Embedding Layer: Converts text sequences into dense vectors of fixed size.
- Conv1D Layer: Applies 1D convolutions to the embedded sequences.
- GlobalMaxPooling1D Layer: Reduces the dimensionality by taking the maximum value from each feature map.
- Dropout Layers: Adds regularization to prevent overfitting.
- Dense Output Layer: Outputs the classification in 34 categories using softmax activation.

### Training

The model is trained using the following hyperparameters:

- **Epochs:** 5
- **Batch size:** 45
- **Loss function:** Categorical cross-entropy
- **Optimizer:** Adam

### Callbacks

The model uses the following callbacks for optimization:

- Early Stopping
- Reduce Learning Rate on Plateau

## Results Visualization

The model's accuracy and loss during training are plotted using matplotlib for both the training and validation datasets.

## Prediction on Test Data

The model is used to predict the categories of a test dataset. The output is saved to a CSV file (`out.csv`).
