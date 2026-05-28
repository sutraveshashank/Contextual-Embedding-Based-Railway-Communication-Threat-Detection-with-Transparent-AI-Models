# 🚆 Contextual Embedding-Based Railway Communication Threat Detection with Transparent AI Models

## 📌 Overview

Railway communication systems continuously generate large volumes of signaling data, operational logs, and message transmissions. Any anomaly or security vulnerability in these communication streams can result in operational failures, communication disruption, and safety risks.

This project presents an intelligent and interpretable railway communication threat detection system using **Sentence-BERT (SBERT)** contextual embeddings and transparent machine learning models such as **RuleFit**, **DNDT**, and **Decision Tree CCP**.

The system is implemented as a Django-based web application capable of performing real-time classification of communication messages into:

- ✅ Secure
- ❌ Not Secure

The proposed RuleFit-based approach improves:
- Threat detection accuracy
- Interpretability
- Scalability
- Real-time monitoring capability

making it highly suitable for safety-critical railway environments.

---

# 🎯 Objectives

- Build a Django-based railway communication security monitoring platform.
- Use SBERT embeddings for semantic understanding of communication messages.
- Detect communication anomalies using interpretable AI models.
- Provide real-time threat detection and prediction.
- Improve railway operational safety through Explainable AI (XAI).

---

# ✨ Key Features

- 🔍 AI-powered railway communication threat detection
- 🧠 Contextual embedding generation using SBERT
- 📊 Explainable AI-based transparent predictions
- ⚡ Real-time monitoring and classification
- 📈 Dashboard with confidence score visualization
- 🛡️ Automated secure communication analysis
- 🔄 Semantic similarity analysis of messages

---

# 🏗️ System Architecture

```text
Railway Communication Data
            ↓
      Data Preprocessing
            ↓
   Contextual Embedding Generation
            ↓
      Feature Extraction
            ↓
     AI/ML Classification Model
            ↓
   Threat Detection & Prediction
            ↓
 Explainable AI Visualization
```

---

# 🔄 Traditional System vs Proposed System

## 🏛️ Traditional System

Traditional railway communication monitoring systems rely mainly on manual inspections and predefined rules for detecting anomalies.

### 🔧 How Traditional Systems Work
- Manual monitoring of communication equipment
- Dependence on predefined rules
- Use of historical logs and alarms
- Human operator-based fault identification

### ❌ Limitations of Traditional Systems
- Reactive detection after failures occur
- Cannot detect hidden or complex anomalies
- High dependency on human judgment
- Poor scalability for real-time large-scale data
- Increased downtime and maintenance costs
- Lack of semantic understanding

---

## 🚀 Proposed System

The proposed system uses contextual embeddings and Explainable AI to intelligently detect communication threats in railway systems.

### ⚙️ Proposed Methodology
- SBERT-based semantic embedding generation
- RuleFit and interpretable ML classification
- Real-time communication monitoring
- Transparent and explainable predictions
- Dashboard-based threat visualization

### 🧠 Core Technologies
- Sentence-BERT (SBERT)
- RuleFit Classifier
- Deep Neural Decision Tree (DNDT)
- Django Framework
- Explainable AI (XAI)

### ✅ Advantages of Proposed System
- High detection accuracy (~99.4%)
- Real-time anomaly detection
- Explainable AI predictions
- Automated monitoring
- Reduced communication failures
- Better scalability and reliability

---


# 🧹 Data Preprocessing & Feature Extraction

- Normalization of numerical metrics
- SBERT semantic embedding generation
- Variance threshold filtering
- Label encoding
- Feature extraction and dimensionality reduction

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Framework
- Django 4.2.7

## Database
- MySQL 8.0+

## Libraries
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Sentence-Transformers
- RuleFit
- SHAP / LIME

---

# 📂 Project Structure

```text
Contextual-Embedding-Based-Railway-Communication-Threat-Detection-with-Transparent-AI-Models/
│
├── Dataset/
├── Models/
├── notebooks/
├── preprocessing/
├── training/
├── evaluation/
├── results/
├── app/
├── requirements.txt
├── README.md
└── main.py
```

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/sutraveshashank/Contextual-Embedding-Based-Railway-Communication-Threat-Detection-with-Transparent-AI-Models.git
```

## 2️⃣ Navigate to Project Directory

```bash
cd Contextual-Embedding-Based-Railway-Communication-Threat-Detection-with-Transparent-AI-Models
```

## 3️⃣ Create Virtual Environment

```bash
python -m venv venv
```

## 4️⃣ Activate Environment

### Windows
```bash
venv\Scripts\activate
```

### Linux / Mac
```bash
source venv/bin/activate
```

## 5️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Usage

Run the application:

```bash
python main.py
```

For Django server:

```bash
python manage.py runserver
```

---

# 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- True Positive Rate
- False Positive Rate

---

# 🔍 Explainable AI (XAI)

This project integrates Explainable AI techniques to provide transparent predictions.

### Benefits of XAI
- Improves trust in AI systems
- Explains prediction decisions
- Helps operators understand anomalies
- Supports safer railway communication monitoring

---

# 📄 Research Publication

## 📚 Publication Details

**Title:**  
*Contextual Embedding-Based Railway Communication Threat Detection with Transparent AI Models*

**Status:** Published Research Work

🔗 **Abstract Link:** https://ajaccm.com/journal/index.php/ajaccm/article/view/503

---

# 🚀 Future Scope

- Cybersecurity attack detection
- IoT edge deployment
- Multi-language communication support
- Advanced transformer integration
- Real-time edge intelligence systems

---

# 👨‍💻 Contributors

- Suthrave Shashank
- D Sai Varun
- M Tarun Kumar

---


# 🌟 GitHub Repository

https://github.com/sutraveshashank/Contextual-Embedding-Based-Railway-Communication-Threat-Detection-with-Transparent-AI-Models
