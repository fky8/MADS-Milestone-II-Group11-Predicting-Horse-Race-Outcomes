# baseline_model.py
# This script builds a baseline binary classification model to predict if a horse finishes in the top 3.

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer


# Function to load race data, merge with jockey stats and weather data, and preprocess for modeling
def load_and_merge_data(
    race_path: str,
    stats_path: str,
    names_path: str,
    weather_path: str,
) -> pd.DataFrame:
    """Load race results, merge jockey cumulative stats, and add weather data."""
    # Load main race dataset
    race_df = pd.read_csv(race_path)
    # Drop unnecessary columns that may induce leakage
    race_df = race_df.drop(columns=['Total Stakes', 'Age', 'No. of 1-2-3-Starts',
                                    'No. of starts in past 10 race meetings',
                                    'Current Stable Location (Arrival Date)',
                                    'Last Rating For Retired','Start of Season Rating',
                                    'Current Rating','Season Stakes'], errors='ignore')
    # Ensure 'Date' is in datetime format
    race_df['Date'] = pd.to_datetime(race_df['Date'], errors='coerce')

    # Merge in daily weather data by matching on 'Date' and 'Course'
    weather_df = pd.read_csv(weather_path)
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], format='%Y/%m/%d', errors='coerce')
    race_df = race_df.merge(weather_df, on=['Date', 'Course'], how='left')

    # Map jockey names to their unique identifiers
    names_df = pd.read_excel(names_path)
    race_df = race_df.merge(names_df, left_on='Jockey', right_on='jockey_name', how='left')

    # Load and clean jockey stats, ensuring valid dates and jockey IDs
    stats_df = pd.read_csv(stats_path)
    stats_df['race_date'] = pd.to_datetime(stats_df['race_date'], errors='coerce')
    stats_df = stats_df.dropna(subset=['race_date', 'jockey_id'])
    stats_df = stats_df.sort_values(['jockey_id', 'race_date']).reset_index(drop=True)

    # Perform a per-jockey as-of merge to get latest stats before each race
    race_df = race_df.dropna(subset=['Date', 'jockey_id'])
    merged_parts = []
    for jid, group in race_df.groupby('jockey_id', dropna=False):  # include NaN groups
        if pd.notna(jid):
            stats_j = stats_df[stats_df['jockey_id'] == jid]
            if not stats_j.empty:
                merged_group = pd.merge_asof(
                    group.sort_values('Date'),
                    stats_j.sort_values('race_date'),
                    left_on='Date',
                    right_on='race_date',
                    direction='backward',
                    allow_exact_matches=False,
                )
            else:
                merged_group = group
        else:
            merged_group = group  # rows with unknown jockey_id keep NaN stats
        merged_parts.append(merged_group)

    merged = pd.concat(merged_parts, ignore_index=True)

    # Convert win rate columns from string percentages to numeric fractions
    percent_cols = [
        'first_place_win_rate',
        'second_place_win_rate',
        'third_place_win_rate',
        'total_win_rate',
    ]
    for col in percent_cols:
        if col in merged.columns:
            merged[col] = merged[col].str.rstrip('%').astype(float) / 100.0

    # Convert column to numeric format, coercing errors to NaN
    merged['Win Odds'] = pd.to_numeric(merged['Win Odds'], errors='coerce')
    # Convert column to numeric format, coercing errors to NaN
    merged['Handicap'] = pd.to_numeric(merged['Handicap'], errors='coerce')
    # Convert column to numeric format, coercing errors to NaN
    merged['Dr.'] = pd.to_numeric(merged['Dr.'], errors='coerce')
    # Convert column to numeric format, coercing errors to NaN
    merged['DistanceMeter'] = pd.to_numeric(merged['DistanceMeter'], errors='coerce')

    # Create binary target label: 1 if placing is 1st to 3rd, else 0
    merged['label'] = (pd.to_numeric(merged['Placing'], errors='coerce') <= 3).astype(int)
    return merged


# Function to split data into train/validation sets, build a pipeline, and evaluate a logistic regression model
def build_and_evaluate(df: pd.DataFrame) -> None:
    # Drop rows with missing date
    # df = df.dropna(subset=['Date'])
    # Sort data by date to simulate time-aware train/test split
    df = df.sort_values('Date')

    # Select features excluding label and original placing column
    feature_cols = [c for c in df.columns if c not in {'label', 'Placing'}]
    # Convert datetime columns to ordinal numbers for modeling
    datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    for col in datetime_cols:
        tmp = pd.to_datetime(df[col], errors='coerce')
        df[col + '_ordinal'] = tmp.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
        feature_cols.append(col + '_ordinal')
        feature_cols.remove(col)

    # Split the dataset by date into 80% training and 20% validation sets
    split_date = df['Date'].quantile(0.8)
    train_df = df[df['Date'] < split_date]
    val_df = df[df['Date'] >= split_date]

    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_val = val_df[feature_cols]
    y_val = val_df['label']

    # Separate categorical and numerical columns for preprocessing
    cat_cols = [c for c in feature_cols if X_train[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Define preprocessing steps for numeric columns
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    # Define preprocessing steps for categorical columns
    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )

    # Combine preprocessing steps into a single transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ]
    )

    # Build modeling pipeline: preprocessing + logistic regression
    clf = Pipeline(steps=[('preprocess', preprocessor), ('model', LogisticRegression(max_iter=1000))])

    # Train the model
    clf.fit(X_train, y_train)
    # Predict labels on validation data
    preds = clf.predict(X_val)
    # Predict probabilities for positive class
    probas = clf.predict_proba(X_val)[:, 1]

    # Output classification performance and ROC AUC score
    print(classification_report(y_val, preds))
    print('ROC AUC:', roc_auc_score(y_val, probas))


# Run the data loading and model building pipeline
if __name__ == "__main__":
    data = load_and_merge_data(
        "data/Race_comments_gear_horse_competitors_2019_2025.csv",
        "data/jockey/jockey_win_stats_cumsum.csv",
        "data/jockey/jockey_names.xlsx",
        "data/weather_data/race_date_with_weather.csv",
    )
    data.to_csv("dataset_to_train.csv", index=False)
    build_and_evaluate(data)