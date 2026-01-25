import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
import html
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase (sempre fondamentale per generalizzare)
    text = text.lower()
    
    text = html.unescape(text)
    
    text = re.sub(r'[<>/\\\.=":_\-]', ' ', text)
    
    stop_garbage = r'\b(http|https|www|com|href|src|jpg|png|html|height|width|border)\b'
    text = re.sub(stop_garbage, ' ', text)
    
    text = re.sub(r'\d+', 'NUM', text)
    
    text = re.sub(r'\b[a-z]\b', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def text_cleaning_pipeline(df):
    
    df['title'] = df['title'].fillna('')
    df['article'] = df['article'].fillna('')
    
    df['article'] = df['article'].apply(clean_text)
    df['title'] = df['title'].apply(clean_text)

    df['text_content'] = df['title']+ ' ' + df['article']

    return df

def remove_duplicates(train_df):

    group_cols = ['text_content']

    def get_majority_or_tie(x):
        modes = x.mode()
        if len(modes) == 1:
            return modes[0]
        else:
            return -1

    voted_labels = train_df.groupby(group_cols)['label'].transform(get_majority_or_tie)
    train_df['label'] = voted_labels
    train_df = train_df[train_df['label'] != -1]
    train_df = train_df.drop_duplicates(subset=group_cols, keep='first')

    group_cols = ['article']
    mask_process = train_df['article'].str.strip() != ''
    df_to_process = train_df[mask_process].copy()
    df_ignored = train_df[~mask_process].copy()

    if not df_to_process.empty:
        voted_labels = df_to_process.groupby(group_cols)['label'].transform(get_majority_or_tie)
        df_to_process['label'] = voted_labels
        df_to_process = df_to_process[df_to_process['label'] != -1]
        df_to_process = df_to_process.drop_duplicates(subset=group_cols, keep='first')

    
    train_df = pd.concat([df_to_process, df_ignored], ignore_index=True)
    return train_df


def main():
    
    print("Loading data...")

    train_df = pd.read_csv("development.csv", na_values=['\\N', '0000-00-00 00:00:00', '  '], encoding='utf-8')
    eval_df = pd.read_csv("evaluation.csv", na_values=['\\N', '0000-00-00 00:00:00', '  '], encoding='utf-8')
    
    print("Data loaded successfully.")
    
    print("Cleaning text data...")
    train_df = text_cleaning_pipeline(train_df)
    eval_df = text_cleaning_pipeline(eval_df)
    print("Text data cleaned.")
    
    print("Removing duplicates...")
    train_df = remove_duplicates(train_df)
    print("Duplicates removed. Training data size:", len(train_df))
    
    text_transformer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    lowercase=True,
    strip_accents='unicode',
    ngram_range=(1,3),
    max_df=0.5,
    min_df=2
    )

    cat_transformer = OneHotEncoder(handle_unknown='ignore')

    hour_transformer = Pipeline([
        ('imputer',SimpleImputer(strategy='mean')),
        ])
    source_pipe = Pipeline([
        ('encoder', OneHotEncoder(max_categories=150, handle_unknown='infrequent_if_exist')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', text_transformer, 'text_content'),
            ('cat', cat_transformer, ['page_rank']),
            ('hour', hour_transformer, ['hour_cos','hour_sin']),
            ('cat_source', source_pipe, ['source'])    
            ],
        remainder='passthrough'
    )
    train_df['day_of_week'] = pd.to_datetime(train_df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S').dt.dayofweek
    eval_df['day_of_week'] = pd.to_datetime(eval_df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S').dt.dayofweek

    train_df['is_weekend'] = train_df['day_of_week'].apply(lambda x: 1 if x >=5 else 0)
    train_df['timestamp_missing'] = train_df['day_of_week'].apply(lambda x: 1 if pd.isna(x) else 0)
    
    eval_df['is_weekend'] = eval_df['day_of_week'].apply(lambda x: 1 if x >=5 else 0)
    eval_df['timestamp_missing'] = eval_df['day_of_week'].apply(lambda x: 1 if pd.isna(x) else 0)

    train_df['hour_of_day'] = pd.to_datetime(train_df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S').dt.hour
    eval_df['hour_of_day'] = pd.to_datetime(eval_df['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S').dt.hour
    
    train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour_of_day'] / 24)
    train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour_of_day'] / 24)

    eval_df['hour_sin'] = np.sin(2 * np.pi * eval_df['hour_of_day'] / 24)
    eval_df['hour_cos'] = np.cos(2 * np.pi * eval_df['hour_of_day'] / 24)

    X_eval = eval_df[['text_content', 'source','page_rank','is_weekend','timestamp_missing','hour_sin','hour_cos']]

    X= train_df[['text_content', 'source','page_rank','is_weekend','timestamp_missing','hour_sin','hour_cos']]
    y = train_df['label']
    
    classifier = LinearSVC(class_weight='balanced', random_state=42, C=0.3)
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    print("Training model...")
    model_pipeline.fit(X, y)
    print("Model trained.")
    print("Predicting on evaluation data...")
    eval_predictions = model_pipeline.predict(X_eval)
    print("Predictions completed.")
    
    submission_df = pd.DataFrame({
        'Id': eval_df['Id'],
        'Predicted': eval_predictions
    })
    
    submission_df.to_csv('submission.csv', index=False)
    print("Submission file created: submission.csv")
    
main()