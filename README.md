# 📱 SMS Spam Classification using NLP & Machine Learning
![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)
![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-blueviolet)

An intelligent system that classifies SMS messages as **ham** (legitimate) or **spam** (malicious/unwanted) using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

This project aims to improve mobile communication security by accurately filtering unsolicited messages that often carry threats like phishing, scams, and malware.

---

## 🧭 Table of Contents
1. [📘 Introduction & Problem Statement](#📘-introduction--problem-statement)
2. [📂 Dataset](#📂-dataset)
3. [🔧 Setup & Dependencies](#🔧-setup--dependencies)
4. [📥 Data Loading & Initial Exploration](#📥-data-loading--initial-exploration)
5. [🧼 Data Preprocessing](#🧼-data-preprocessing)
6. [📊 Exploratory Data Analysis](#📊-exploratory-data-analysis)
7. [⚙️ Feature Engineering](#⚙️-feature-engineering)
8. [🤖 Model Selection & Training](#🤖-model-selection--training)
9. [🧪 Model Evaluation](#🧪-model-evaluation)
10. [💾 Saving the Final Model](#💾-saving-the-final-model)
11. [🔮 Conclusion & Future Work](#🔮-conclusion--future-work)

---

## 📘 Introduction & Problem Statement

### 🔍 Project Goal
Build a machine learning model to classify SMS messages into two categories:
- **Ham**: Legitimate messages
- **Spam**: Unsolicited, fraudulent, or malicious messages

### 🚨 The Problem
SMS spam remains a global issue:
- Financial scams
- Privacy violations
- Poor user experience

Keyword-based filters fail to adapt to evolving spam tactics, causing:
- **False positives**: Good messages marked as spam
- **False negatives**: Spam sneaks through

### 💡 Why This Project Matters
- 🔐 **User Protection** from phishing and scams
- 📱 **Enhanced User Experience** through cleaner inboxes
- 🌐 **Resource Optimization** for networks
- 🧠 **Adaptability** through retraining on new data
- 🛡️ **Security Foundation** for broader mobile defense systems

---

## 📂 Dataset

- **Source**: [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size**: 5,574 SMS messages labeled as "ham" or "spam"
- **Format**: Tab-separated values with two fields:
  - `label`: spam or ham
  - `message`: actual SMS text

---

## 🔧 Setup & Dependencies

To get started locally or on Google Colab:

```bash
# Clone this repo
git clone https://github.com/KalihoseMigisha/sms-spam-classifier.git
cd sms-spam-classifier

# Install dependencies
pip install -r requirements.txt
```

### 📦 Main Libraries Used:
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `nltk` / `spaCy`
- `joblib` or `pickle` (for model saving)

---

## 📥 Data Loading & Initial Exploration

- Loaded dataset into pandas
- Checked for:
  - Missing values
  - Message length distribution
  - Class imbalance (spam vs ham)

---

## 🧼 Data Preprocessing

- Lowercasing
- Removing special characters, punctuation
- Tokenization
- Stopword removal
- Lemmatization/Stemming

---

## 📊 Exploratory Data Analysis

- Visualized:
  - Class distribution
  - Most common spam words
  - Word cloud for ham vs spam

---

## ⚙️ Feature Engineering

- Text Vectorization using:
  - **TF-IDF**
  - **CountVectorizer**
- Optional: Word embeddings (e.g., Word2Vec, GloVe)
- Engineered features:
  - Message length
  - Punctuation count
  - Capital letter usage

---

## 🤖 Model Selection & Training

Tested multiple models:
- Logistic Regression ✅
- Multinomial Naive Bayes ✅
- Random Forest
- Support Vector Machine (SVM)
- LSTM / GRU (optional for deep learning)

Used **train-test split** to evaluate.

---

## 🧪 Model Evaluation

Metrics used:
- Accuracy
- Precision
- Confusion Matrix
- Recall (Try this too: Actual Positive Correctly Predicted)
- F1 Score (Try this too: Harmonic mean of Sensitivity and Recall)
- ROC-AUC Curve(Try this too: FP Rate vs FN Rate)

Visuals and performance summaries available in `/outputs`.

---

## 💾 Saving the Final Model

Used `pickle` to serialize the best-performing model:
```python
import pickle
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
```

Also saved the vectorizer for future predictions.

---

## 🔮 Conclusion & Future Work

### ✅ Key Takeaways
- Achieved high accuracy on test set
- Naive Bayes and Logistic Regression performed best for baseline
- Preprocessing and vectorization were crucial

### 🔭 Future Enhancements
- Incorporate deep learning models (e.g., LSTM)
- Add real-time classification API
- Expand dataset to include multilingual SMS
- Integrate with email or messaging clients

---

## 📎 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- Python, scikit-learn, NLTK, and all open-source contributors

---

> 🚀 **Built with passion for research, security, and machine learning.**
>  
> 💼 Maintained by [KalihoseMigisha](https://github.com/KalihoseMigisha)
