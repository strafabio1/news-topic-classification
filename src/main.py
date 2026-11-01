import pandas as pd
import numpy as np
import html
import re
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def clean_text(text):
    if not isinstance(text, str):
        return ""
    
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

    df['text_content'] = df['title'] + ' ' + df['article']
    return df

def remove_duplicates(train_df):
    train_df = train_df[train_df['text_content'] != " "].copy()
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

def extract_time_features(df):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    col_name = df.columns[0]
    dt = pd.to_datetime(df[col_name], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    
    day_of_week = dt.dt.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)
    
    timestamp_missing = dt.isna().astype(int)
    
    hour = dt.dt.hour
    hour_sin = np.sin(2 * np.pi * hour / 24).fillna(0)
    hour_cos = np.cos(2 * np.pi * hour / 24).fillna(0)
    
    return pd.concat([
        hour_sin.rename('hour_sin'), 
        hour_cos.rename('hour_cos'),
        is_weekend.rename('is_weekend'), 
        timestamp_missing.rename('timestamp_missing')
    ], axis=1)

def main():
    np.random.seed(42)
    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "development.csv", na_values=['\\N', '0000-00-00 00:00:00', '  '], encoding='utf-8')
    eval_df = pd.read_csv(DATA_DIR / "evaluation.csv", na_values=['\\N', '0000-00-00 00:00:00', '  '], encoding='utf-8')
    print("Data loaded successfully.")
    
    print("Cleaning text data...")
    train_df = text_cleaning_pipeline(train_df)
    eval_df = text_cleaning_pipeline(eval_df)
    print("Text data cleaned.")
    
    print("Removing duplicates...")
    train_df = remove_duplicates(train_df)
    train_df=train_df.dropna(subset=['source'])
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

    time_transformer = Pipeline([
        ('time_extraction', FunctionTransformer(extract_time_features, validate=False)),
    ])

    source_pipe = Pipeline([
        ('encoder', OneHotEncoder(max_categories=150, handle_unknown='infrequent_if_exist')),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('txt', text_transformer, 'text_content'),
            ('cat', cat_transformer, ['page_rank']),
            ('cat_source', source_pipe, ['source']),
            ('time', time_transformer, ['timestamp']), 
        ],
        remainder='passthrough'
    )

    X = train_df[['text_content', 'source', 'page_rank', 'timestamp']]
    y = train_df['label']
    
    X_eval = eval_df[['text_content', 'source', 'page_rank', 'timestamp']]

    classifier = LinearSVC(class_weight='balanced', random_state=42, C=0.3)
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])

    print("Running Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model_pipeline, X, y, cv=skf, scoring='f1_macro', n_jobs=-1)
    print(f"CV F1 Macro Mean: {cv_scores.mean():.4f}")
    print(f"CV F1 Macro Std: {cv_scores.std():.4f}")
    
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
    
    submission_df.to_csv(ROOT / 'submission.csv', index=False)
    print("Submission file created: submission.csv")

if __name__ == "__main__":
    main()