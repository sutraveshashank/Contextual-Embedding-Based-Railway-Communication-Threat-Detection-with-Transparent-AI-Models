# 🚆 Anomaly Detection in Railway Control Communication

### 📌 Overview

This project focuses on detecting anomalies in railway communication systems using Natural Language Processing (NLP) and interpretable machine learning models. Railway communication is safety-critical, and any abnormal or malicious message can lead to serious failures.

The system analyzes communication logs and classifies them as Secure or Not Secure, ensuring reliability, transparency, and safety in railway operations.



### 🎯 Aim

The main objective of this project is to build an intelligent anomaly detection system that:

Identifies insecure or abnormal railway communication messages <br>
Uses semantic understanding (SBERT) instead of traditional keyword-based methods.    
Provides interpretable results suitable for safety-critical environments


### 🧠 Key Features
🔍 Semantic analysis using Sentence-BERT (SBERT) <br>
🤖 Interpretable ML models (RuleFit, Decision Trees, DNDT) <br>
📊 High performance (93.3% F1-score, 0.96 AUC)  
🧾 Explainable predictions with IF-THEN rules <br>
🌐 Interactive Django-based web interface <br>
⚡ Optimized with embedding caching for faster inference




### 🛠️ Tech Stack
Programming Language: Python <br>
Framework: Django
Libraries: Scikit-learn, Sentence-Transformers, NLTK <br>
Tools: Jupyter Notebook




### ⚡ Working
The system accepts railway communication messages as input. <br>
Applies NLP preprocessing (tokenization, stopword removal, lemmatization). <br>
Converts text into embeddings using Sentence-BERT. <br>
Uses interpretable ML models (RuleFit, DNDT, Decision Trees) for classification. <br>
Outputs whether the message is Secure or Not Secure. <br>
Displays results in a Django web interface along with explainable insights.



### 📊 Results
✅ Accuracy: 93.4% <br>
✅ F1 Score: 93.3% <br>
✅ AUC: 0.96 <br>
📌 RuleFit provided the best performance with interpretable rules
