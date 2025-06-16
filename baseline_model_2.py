import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer


def load_and_merge_data(race_path: str, weather_path: str) -> pd.DataFrame:
    """Load race results and merge daily weather information."""
    race_df = pd.read_csv(race_path)
    race_df['Date'] = pd.to_datetime(race_df['Date'], errors='coerce')

    weather_df = pd.read_csv(weather_path)
    weather_df['Date'] = pd.to_datetime(weather_df['Date'], errors='coerce')

    # join on both date and course to ensure alignment
    merged = race_df.merge(weather_df, on=['Date', 'Course'], how='left')

    merged['Win Odds'] = pd.to_numeric(merged['Win Odds'], errors='coerce')
    merged['Handicap'] = pd.to_numeric(merged['Handicap'], errors='coerce')
    merged['Dr.'] = pd.to_numeric(merged['Dr.'], errors='coerce')
    merged['DistanceMeter'] = pd.to_numeric(merged['DistanceMeter'], errors='coerce')

    merged['label'] = (pd.to_numeric(merged['Placing'], errors='coerce') <= 3).astype(int)
    return merged


def build_and_evaluate(df: pd.DataFrame) -> None:
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    # limit size for quicker baseline demonstration
    df = df.head(2000)

    # use all available columns except the label and target
    feature_cols = [c for c in df.columns if c not in {'label', 'Placing'}]
    # Convert datetime to numeric ordinal for modeling
    datetime_cols = [c for c in feature_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    for col in datetime_cols:
        tmp = pd.to_datetime(df[col], errors='coerce')
        df[col + '_ordinal'] = tmp.map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
        feature_cols.append(col + '_ordinal')
        feature_cols.remove(col)

    split_date = df['Date'].quantile(0.8)
    train_df = df[df['Date'] < split_date]
    val_df = df[df['Date'] >= split_date]

    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_val = val_df[feature_cols]
    y_val = val_df['label']

    cat_cols = [c for c in feature_cols if X_train[c].dtype == 'object']
    num_cols = [c for c in feature_cols if c not in cat_cols]

    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols),
        ]
    )

    clf = Pipeline(steps=[('preprocess', preprocessor), ('model', LogisticRegression(max_iter=1000))])

    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    probas = clf.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, preds))
    print('ROC AUC:', roc_auc_score(y_val, probas))


if __name__ == "__main__":
    data = load_and_merge_data(
        "data/Race_comments_gear_horse_competitors_2019_2025.csv",
        "data/weather_data/race_date_with_weather.csv",
    )
    data.to_csv("dataset_2.csv", index=False)
    build_and_evaluate(data)