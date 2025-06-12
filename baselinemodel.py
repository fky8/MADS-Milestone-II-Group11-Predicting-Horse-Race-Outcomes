import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def load_and_merge_data(race_path: str, stats_path: str, names_path: str) -> pd.DataFrame:
    """Load race results and merge jockey cumulative stats."""
    race_df = pd.read_csv(race_path)
    race_df['Date'] = pd.to_datetime(race_df['Date'], errors='coerce')

    # Map jockey names to identifiers
    names_df = pd.read_excel(names_path)
    race_df = race_df.merge(names_df, left_on='Jockey', right_on='jockey_name', how='left')

    stats_df = pd.read_csv(stats_path)
    stats_df['race_date'] = pd.to_datetime(stats_df['race_date'], errors='coerce')
    stats_df = stats_df.dropna(subset=['race_date', 'jockey_id'])
    stats_df = stats_df.sort_values(['jockey_id', 'race_date']).reset_index(drop=True)

    # merge_asof per jockey to get stats from latest date not after the race
    race_df = race_df.dropna(subset=['Date', 'jockey_id'])
    merged_parts = []
    for jid, group in race_df.groupby('jockey_id'):
        stats_j = stats_df[stats_df['jockey_id'] == jid]
        if not stats_j.empty:
            merged_group = pd.merge_asof(
                group.sort_values('Date'),
                stats_j.sort_values('race_date'),
                left_on='Date',
                right_on='race_date',
                direction='backward'
            )
        else:
            merged_group = group
        merged_parts.append(merged_group)
    merged = pd.concat(merged_parts, ignore_index=True)

    # convert percentage strings to numeric rates
    percent_cols = [
        'first_place_win_rate',
        'second_place_win_rate',
        'third_place_win_rate',
        'total_win_rate',
    ]
    for col in percent_cols:
        if col in merged.columns:
            merged[col] = merged[col].str.rstrip('%').astype(float) / 100.0

    merged['Win Odds'] = pd.to_numeric(merged['Win Odds'], errors='coerce')
    merged['Handicap'] = pd.to_numeric(merged['Handicap'], errors='coerce')
    merged['Dr.'] = pd.to_numeric(merged['Dr.'], errors='coerce')
    merged['DistanceMeter'] = pd.to_numeric(merged['DistanceMeter'], errors='coerce')

    merged['label'] = (pd.to_numeric(merged['Placing'], errors='coerce') <= 3).astype(int)
    return merged


def build_and_evaluate(df: pd.DataFrame) -> None:
    feature_cols_cat = ['Course', 'Race type', 'Going']
    feature_cols_num = [
        'DistanceMeter',
        'Handicap',
        'Dr.',
        'Win Odds',
        'first_place_win_rate',
        'second_place_win_rate',
        'third_place_win_rate',
        'total_win_rate',
    ]

    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date')
    df = df.dropna(subset=feature_cols_cat + feature_cols_num)
    split_date = df['Date'].quantile(0.8)
    train_df = df[df['Date'] < split_date]
    val_df = df[df['Date'] >= split_date]

    X_train = train_df[feature_cols_cat + feature_cols_num]
    y_train = train_df['label']
    X_val = val_df[feature_cols_cat + feature_cols_num]
    y_val = val_df['label']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), feature_cols_cat),
            ('num', 'passthrough', feature_cols_num),
        ]
    )

    clf = Pipeline(
        steps=[('preprocess', preprocessor), ('model', LogisticRegression(max_iter=1000))]
    )

    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    probas = clf.predict_proba(X_val)[:, 1]

    print(classification_report(y_val, preds))
    print('ROC AUC:', roc_auc_score(y_val, probas))


if __name__ == "__main__":
    data = load_and_merge_data(
        "data/Race_comments_gear_horse_competitors_2019_2025.csv",
        "data/jockey/jockey_win_stats_cumsum.csv",
        "data/jockey/jockey_names.xlsx",
    )
    build_and_evaluate(data)
