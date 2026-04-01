# ===============================
# Django Imports
# ===============================
from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
import pymysql
import pandas as pd
import os
import joblib
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import torch
from transformers import AutoTokenizer, AutoModel

# ===============================
# NLTK Data Setup
# ===============================
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ===============================
# Global Configuration
# ===============================
MODEL_DIR = os.path.join(settings.BASE_DIR, "userapp", "model")
os.makedirs(MODEL_DIR, exist_ok=True)

CUSTOM_LABEL_MAP = {'Target': ["Not Secure", "Secure"]}

# ===============================
# Database Connection
# ===============================
def get_connection():
    return pymysql.connect(
        host='127.0.0.1',
        port=3306,
        user='root',
        password='Password@123',
        database='brandusers',
        charset='utf8'
    )

# ===============================
# UI Views
# ===============================
def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        address = request.POST.get('address')
        username = request.POST.get('username')
        password = request.POST.get('password')

        con = get_connection()
        with con:
            cur = con.cursor()
            sql = """
                INSERT INTO users (name, phone, email, address, username, password)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            cur.execute(sql, (name, phone, email, address, username, password))
            con.commit()

        messages.success(request, "Registration successful! You can now login.")
    return render(request, 'register.html')

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        con = get_connection()
        with con:
            cur = con.cursor()
            sql = "SELECT * FROM users WHERE username=%s AND password=%s"
            cur.execute(sql, (username, password))
            user = cur.fetchone()

        if user:
            return redirect('predict_from_file')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')

# ===============================
# Text Preprocessing
# ===============================
def preprocess_data(df):
    """Replicates the preprocessing logic used in Jupyter notebook."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        text = str(text).lower()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalnum() and t not in stop_words]
        return ' '.join(tokens)

    text_columns = df.select_dtypes(include='object').columns
    for col in text_columns:
        df[f'processed_{col}'] = df[col].apply(clean_text)
    df.drop(columns=text_columns, inplace=True, errors='ignore')

    processed_cols = [col for col in df.columns if col.startswith('processed_')]
    X_text = df[processed_cols].astype(str).agg(' '.join, axis=1).tolist()
    return X_text

# ===============================
# SBERT Feature Extraction (Cached)
# ===============================
_tokenizer = None
_model = None

def sbert_feature_extraction(texts, model_name='sentence-transformers/paraphrase-mpnet-base-v2', batch_size=32):
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        _model.eval()

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encoded_input = _tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')

        with torch.no_grad():
            model_output = _model(**encoded_input)

        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        embeddings = sum_embeddings / sum_mask

        all_embeddings.append(embeddings.cpu().numpy())

    X = np.vstack(all_embeddings)
    return X

# ===============================
# Feature Extraction Wrapper
# ===============================
def feature_extraction(X_text, model_dir=MODEL_DIR, method='SBERT_paraphrase', is_train=False):
    """Load or generate SBERT embeddings."""
    x_file = os.path.join(model_dir, f"X_{method}.pkl")

    if os.path.exists(x_file):
        X = joblib.load(x_file)
    else:
        X = sbert_feature_extraction(X_text)
        joblib.dump(X, x_file)

    return X

# ===============================
# RuleFit Prediction View
# ===============================
def predict_from_file(request):
    prediction_table = None
    uploaded_filename = None

    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        uploaded_filename = file.name
        df_test = pd.read_csv(file)
        df_result = df_test.copy()

        # Step 1: Preprocess uploaded CSV
        df_test_processed = preprocess_data(df_test)

        # Step 2: Generate or load SBERT features
        features_test = feature_extraction(df_test_processed, method='SBERT_paraphrase', is_train=False)

        # Step 3: Load RuleFit model
        model_path = os.path.join(MODEL_DIR, "Paraphrase SBERT_Target_rulefit_model.pkl")
        if not os.path.exists(model_path):
            messages.error(request, "RuleFit model file not found. Please train it first.")
            return render(request, "predict_file.html")

        model = joblib.load(model_path)

        # Step 4: Predict using RuleFit
        y_pred = model.predict(features_test)
        y_pred = y_pred[:len(df_result)]  # Fix possible mismatch

        mapped_labels = [CUSTOM_LABEL_MAP['Target'][label] for label in y_pred]
        df_result['Predicted_Target'] = mapped_labels

        # Step 5: Convert to HTML table
        prediction_table = df_result.to_html(
            classes="table table-bordered table-striped table-hover",
            index=False
        )

        messages.success(request, f"Predictions generated successfully for {uploaded_filename}")

    return render(request, "predict_file.html", {
        "prediction_table": prediction_table,
        "uploaded_filename": uploaded_filename
    })





# #  Updated Code 


# # ===============================
# # Django Imports
# # ===============================
# from django.shortcuts import render, redirect
# from django.conf import settings
# from django.contrib import messages
# from django.contrib.auth.models import User
# from django.contrib.auth import authenticate, login, logout
# from django.contrib.auth.decorators import login_required

# # ===============================
# # Python & ML Imports
# # ===============================
# import pandas as pd
# import os
# import joblib
# import numpy as np
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import torch
# from transformers import AutoTokenizer, AutoModel

# # ===============================
# # NLTK Setup
# # ===============================
# nltk.download('punkt', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# # ===============================
# # Global Configuration
# # ===============================
# MODEL_DIR = os.path.join(settings.BASE_DIR, "userapp", "model")
# os.makedirs(MODEL_DIR, exist_ok=True)

# CUSTOM_LABEL_MAP = {'Target': ["Not Secure", "Secure"]}

# # ===============================
# # UI Views
# # ===============================
# def home(request):
#     return render(request, 'home.html')


# # ===============================
# # REGISTER VIEW (Secure)
# # ===============================
# def register(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')
#         email = request.POST.get('email')

#         if User.objects.filter(username=username).exists():
#             messages.error(request, "Username already exists.")
#             return render(request, 'register.html')

#         User.objects.create_user(
#             username=username,
#             password=password,
#             email=email
#         )

#         messages.success(request, "Registration successful! Please login.")
#         return redirect('login')

#     return render(request, 'register.html')


# # ===============================
# # LOGIN VIEW (Secure)
# # ===============================
# def login_view(request):
#     if request.method == 'POST':
#         username = request.POST.get('username')
#         password = request.POST.get('password')

#         user = authenticate(request, username=username, password=password)

#         if user is not None:
#             login(request, user)
#             return redirect('predict_from_file')
#         else:
#             messages.error(request, "Invalid username or password.")

#     return render(request, 'login.html')


# # ===============================
# # LOGOUT VIEW
# # ===============================
# def logout_view(request):
#     logout(request)
#     return redirect('login')


# # ===============================
# # Text Preprocessing
# # ===============================
# def preprocess_data(df):
#     lemmatizer = WordNetLemmatizer()
#     stop_words = set(stopwords.words('english'))

#     def clean_text(text):
#         text = str(text).lower()
#         tokens = word_tokenize(text)
#         tokens = [
#             lemmatizer.lemmatize(t)
#             for t in tokens
#             if t.isalnum() and t not in stop_words
#         ]
#         return ' '.join(tokens)

#     text_columns = df.select_dtypes(include='object').columns

#     for col in text_columns:
#         df[f'processed_{col}'] = df[col].apply(clean_text)

#     df.drop(columns=text_columns, inplace=True, errors='ignore')

#     processed_cols = [col for col in df.columns if col.startswith('processed_')]

#     X_text = df[processed_cols].astype(str).agg(' '.join, axis=1).tolist()

#     return X_text


# # ===============================
# # SBERT Feature Extraction (Cached)
# # ===============================
# _tokenizer = None
# _model = None

# def sbert_feature_extraction(texts, model_name='sentence-transformers/paraphrase-mpnet-base-v2', batch_size=32):
#     global _tokenizer, _model

#     if _tokenizer is None or _model is None:
#         _tokenizer = AutoTokenizer.from_pretrained(model_name)
#         _model = AutoModel.from_pretrained(model_name)
#         _model.eval()

#     all_embeddings = []

#     for i in range(0, len(texts), batch_size):
#         batch_texts = texts[i:i + batch_size]

#         encoded_input = _tokenizer(
#             batch_texts,
#             padding=True,
#             truncation=True,
#             return_tensors='pt'
#         )

#         with torch.no_grad():
#             model_output = _model(**encoded_input)

#         token_embeddings = model_output.last_hidden_state
#         attention_mask = encoded_input['attention_mask']
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#         sum_mask = input_mask_expanded.sum(dim=1)
#         embeddings = sum_embeddings / sum_mask

#         all_embeddings.append(embeddings.cpu().numpy())

#     X = np.vstack(all_embeddings)
#     return X


# # ===============================
# # Feature Extraction Wrapper
# # ===============================
# def feature_extraction(X_text, model_dir=MODEL_DIR, method='SBERT_paraphrase'):
#     x_file = os.path.join(model_dir, f"X_{method}.pkl")

#     if os.path.exists(x_file):
#         X = joblib.load(x_file)
#     else:
#         X = sbert_feature_extraction(X_text)
#         joblib.dump(X, x_file)

#     return X


# # ===============================
# # RuleFit Prediction View (Protected)
# # ===============================
# @login_required
# def predict_from_file(request):
#     prediction_table = None
#     uploaded_filename = None

#     if request.method == 'POST' and request.FILES.get('file'):
#         file = request.FILES['file']
#         uploaded_filename = file.name

#         df_test = pd.read_csv(file)
#         df_result = df_test.copy()

#         # Step 1: Preprocess
#         df_test_processed = preprocess_data(df_test)

#         # Step 2: Extract Features
#         features_test = feature_extraction(df_test_processed)

#         # Step 3: Load Model
#         model_path = os.path.join(MODEL_DIR, "Paraphrase SBERT_Target_rulefit_model.pkl")

#         if not os.path.exists(model_path):
#             messages.error(request, "RuleFit model not found.")
#             return render(request, "predict_file.html")

#         model = joblib.load(model_path)

#         # Step 4: Predict
#         y_pred = model.predict(features_test)
#         y_pred = y_pred[:len(df_result)]

#         mapped_labels = [CUSTOM_LABEL_MAP['Target'][label] for label in y_pred]
#         df_result['Predicted_Target'] = mapped_labels

#         prediction_table = df_result.to_html(
#             classes="table table-bordered table-striped table-hover",
#             index=False
#         )

#         messages.success(request, f"Predictions generated successfully for {uploaded_filename}")

#     return render(request, "predict_file.html", {
#         "prediction_table": prediction_table,
#         "uploaded_filename": uploaded_filename
#     })