# Spam Email Detection Using TensorFlow

This repository contains a **Deep Learning project** to automatically detect whether an email is **spam** or **ham (not spam)** using **TensorFlow and Python**.  
The implementation follows a classic NLP pipeline with text preprocessing, tokenization, sequence padding, and an LSTM-based neural network. :contentReference[oaicite:0]{index=0}

---

## Project Overview

Spam emails are unwanted or unsolicited messages that flood users’ inboxes. Detecting spam automatically helps reduce clutter and protects users from potential phishing or malicious emails.

This project:

- Loads and preprocesses email text data.
- Balances the dataset.
- Tokenizes and pads text for deep learning models.
- Builds and trains a TensorFlow model with an LSTM architecture.
- Evaluates model accuracy on test data.
- Visualizes training performance. :contentReference[oaicite:1]{index=1}

---

## Key Features

✔ Text cleaning & preprocessing  
✔ Tokenization and padding  
✔ Embedding + LSTM based deep learning model  
✔ Early stopping & learning rate reduction callbacks  
✔ Model evaluation and accuracy visualization  
✔ Easy-to-run code with clear structure

---

## Repository Structure
.
├── SpamEmails.ipynb # Main Colab notebook with entire pipeline
├── Emails.csv # Raw dataset (spam/ham email labels)
├── README.md # This file
└── requirements.txt # Python dependencies (optional)


---

## Dependencies

Before running the code, make sure you have the following installed:

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib seaborn nltk wordcloud

```
# How It Works (High-Level)
1. Load Dataset

The dataset containing labeled emails (spam or ham) is loaded and inspected.

2. Text Preprocessing

Remove punctuation.

Remove stopwords.

Clean email text.

3. Balance the Dataset

Because spam emails are typically fewer than ham emails, we balance the dataset by oversampling or undersampling to equalize classes.

4. Tokenization & Padding

Convert text into integer sequences (tokens) and make them uniform in length with padding:
```bash
train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
```
5. Model Architecture

We use an LSTM-based TensorFlow model:

Embedding layer to learn word representations

LSTM layer for sequence learning

Dense layers for classification

6. Training

The model is trained with validation using callbacks:

EarlyStopping

ReduceLROnPlateau

7. Evaluation

Evaluate model on test set and visualize training/validation accuracy and loss.

 # Sample Results

After training, the model achieves high accuracy (~97% on test data).
```bash
Test Loss: 0.1202
Test Accuracy: 0.9700
```

# How to Run (Colab)

Open the SpamEmails.ipynb notebook in Google Colab.

Upload the Emails.csv dataset.

Run all cells from top to bottom.

Train the model and view results and plots.

# References

Detecting Spam Emails Using TensorFlow in Python — GeeksforGeeks tutorial.
