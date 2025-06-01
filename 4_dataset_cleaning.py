# %%
import pandas as pd
import numpy as np
import re

def clean_column_names(df):
    return df.loc[:, ~df.columns.str.startswith('Unnamed')]

def standardize_columns(race_df, gear_df, horse_df):
    race_df.rename(columns={'Pla.': 'Placing'}, inplace=True)
    gear_df.rename(columns={'Horse No': 'Horse No.', 'Horse Name': 'Horse'}, inplace=True)
    horse_df.rename(columns={'HorseName': 'Horse'}, inplace=True)
    return race_df, gear_df, horse_df

def extract_indices(df, col='Horse'):
    df['HorseIndex'] = df[col].str.extract(r'\(([^)]+)\)')
    df[col] = df[col].str.replace(r'\s*\([^)]*\)', '', regex=True)
    return df

def parse_dates(df, col='Date', fmt='%Y/%m/%d'):
    df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def convert_int_columns(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df

def split_running_position_columns(df, source_col='Running Position'):
    running_pos_cols = [f'RunningPosition{i}' for i in range(1, 7)]
    df[running_pos_cols] = df[source_col].astype(str).str.split(' ', n=5, expand=True)
    for col in running_pos_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    return df

def split_score_range(val):
    if pd.isna(val) or str(val).strip() == '':
        return pd.Series({'MinScore': np.nan, 'MaxScore': np.nan})
    val = str(val).strip()
    match = re.match(r'^(\d+)\s*-\s*(\d+)$', val)
    if match:
        max_score = int(match.group(1))
        min_score = int(match.group(2))
        return pd.Series({'MinScore': min_score, 'MaxScore': max_score})
    match = re.match(r'^(\d+)\+$', val)
    if match:
        min_score = int(match.group(1))
        return pd.Series({'MinScore': min_score, 'MaxScore': np.nan})
    match = re.match(r'^(\d+)$', val)
    if match:
        score = int(match.group(1))
        return pd.Series({'MinScore': score, 'MaxScore': score})
    return pd.Series({'MinScore': np.nan, 'MaxScore': np.nan})

def main():
    # 1. Read data
    race_df = pd.read_csv("RacePlaceData_2010_2025.csv")
    horse_df = pd.read_csv("hkjc_horse_profiles.csv")
    gear_df = pd.read_csv("comments_2010_to_2025_combined.csv")

    # 2. Clean column names
    race_df = clean_column_names(race_df)
    horse_df = clean_column_names(horse_df)
    gear_df = clean_column_names(gear_df)

    # 3. Standardize columns
    race_df, gear_df, horse_df = standardize_columns(race_df, gear_df, horse_df)

    # 4. Extract indices and clean names
    race_df = extract_indices(race_df)
    gear_df = extract_indices(gear_df)

    # 5. Convert columns to correct dtypes
    race_df = convert_int_columns(race_df, ['RaceNumber', 'Horse No.'])
    gear_df = convert_int_columns(gear_df, ['RaceNumber', 'Horse No.'])

    # 6. Parse dates
    race_df = parse_dates(race_df, 'Date')
    gear_df = parse_dates(gear_df, 'Date')

    # 7. Merge and feature engineering (as in your code)
    # Combine race and gear DataFrames by coalescing on keys
    keys = ['Date', 'RaceNumber', 'Horse No.']
    race_gear_df = (
        race_df.set_index(keys)
        .combine_first(gear_df.set_index(keys))
        .reset_index()
    )

    common_idx = (race_df.set_index(keys)
                        .index.intersection(gear_df.set_index(keys).index))
    print(f"Matching rows: {len(common_idx)}")

    # Combine race_gear_df and horse_df by coalescing on Horse (approach 4)
    horse_profile = horse_df.set_index('Horse')
    race_full_df = (
        race_gear_df.set_index('Horse')
        .combine_first(horse_profile)
        .reset_index()
    )

    # Sort race_full_df by Date, RaceNumber, and Placing
    race_full_df = race_full_df.sort_values(by=['Date', 'RaceNumber', 'Placing'])

    # Remove all dollar signs from string values in the DataFrame, excluding column names
    def remove_dollar_signs(val):
        if isinstance(val, str):
            return val.replace('$', '')
        return val
    race_full_df = race_full_df.applymap(remove_dollar_signs)

    # Drop rows where 'Horse' or 'Placing' is missing
    race_full_df = race_full_df.dropna(subset=['Horse', 'Placing'])

    # Drop completely blank rows (all values are NaN)
    race_full_df = race_full_df.dropna(how='all')

    # Add HorseCompetitor1 to HorseCompetitor13 columns
    def assign_competitors(group):
        horse_indices = group['HorseIndex'].tolist()
        for idx, row in group.iterrows():
            competitors = [h for h in horse_indices if h != row['HorseIndex']]
            for i in range(1, 14):
                col_name = f'HorseCompetitor{i}'
                group.at[idx, col_name] = competitors[i-1] if i-1 < len(competitors) else None
        return group

    race_full_df = race_full_df.groupby(['Date', 'RaceNumber'], group_keys=False).apply(assign_competitors)

    # Add JockeyCompetitor1 to JockeyCompetitor13 columns
    def assign_jockeys(group):
        jockeys = group['Jockey'].tolist()
        for idx, row in group.iterrows():
            competitors = [j for j in jockeys if j != row['Jockey']]
            for i in range(1, 14):
                col_name = f'JockeyCompetitor{i}'
                group.at[idx, col_name] = competitors[i-1] if i-1 < len(competitors) else None
        return group

    race_full_df = race_full_df.groupby(['Date', 'RaceNumber'], group_keys=False).apply(assign_jockeys)

    def clean_lbw(val):
        if pd.isna(val) or str(val).strip() == '':
            return float('nan')
        val = str(val).strip()
        if val == '-':
            return 0.0
        match = re.match(r'^(\d+)-(\d+)/(\d+)$', val)
        if match:
            whole = int(match.group(1))
            numerator = int(match.group(2))
            denominator = int(match.group(3))
            return whole + numerator / denominator
        match = re.match(r'^(\d+)/(\d+)$', val)
        if match:
            numerator = int(match.group(1))
            denominator = int(match.group(2))
            return numerator / denominator
        try:
            return float(val)
        except Exception:
            return float('nan')

    if 'LBW' in race_full_df.columns:
        race_full_df['LBW'] = race_full_df['LBW'].apply(clean_lbw).astype(float)

    # Clean and convert Distance column
    if 'Distance' in race_full_df.columns:
        race_full_df = race_full_df.rename(columns={'Distance': 'DistanceMeter'})
        race_full_df['DistanceMeter'] = (
            race_full_df['DistanceMeter']
            .astype(str)
            .str.replace('M', '', regex=False)
            .str.strip()
        )
        race_full_df['DistanceMeter'] = pd.to_numeric(race_full_df['DistanceMeter'], errors='coerce').astype('Int64')

    # Split running position columns
    race_full_df = split_running_position_columns(race_full_df)

    # Split score range if present
    if 'Score range' in race_full_df.columns:
        score_split = race_full_df['Score range'].apply(split_score_range)
        score_split['MinScore'] = pd.to_numeric(score_split['MinScore'], errors='coerce').astype('Int64')
        score_split['MaxScore'] = pd.to_numeric(score_split['MaxScore'], errors='coerce').astype('Int64')
        race_full_df = pd.concat([race_full_df, score_split], axis=1)

    # Rename "Last Rating" to "Last Rating For Retired" if it exists
    if "Last Rating" in race_full_df.columns:
        race_full_df = race_full_df.rename(columns={"Last Rating": "Last Rating For Retired"})

    # Define your desired column order
    desired_order = [
        "Placing","Horse No.","Horse","Jockey","Trainer","Act. Wt.","Declar. Horse Wt.","Dr.","LBW",
        "Win Odds","Date","Course","RaceNumber","Race type","DistanceMeter","Score range","MinScore","MaxScore",
        "Going","Handicap",
        "Course Detail","Time 1","Time 2","Time 3","Time 4","Time 5","Time 6","Sectional Time 1","Sectional Time 2",
        "Sectional Time 3","Sectional Time 4","Sectional Time 5","Sectional Time 6","Running Position","RunningPosition1","RunningPosition2","RunningPosition3",
        "RunningPosition4","RunningPosition5","RunningPosition6","Finish Time","HorseIndex","Country","Age","Colour","Sex","Import Type","Season Stakes","Total Stakes","No. of 1-2-3-Starts",
        "No. of starts in past 10 race meetings","Current Stable Location (Arrival Date)","Import Date","Owner",
        "Current Rating","Last Rating For Retired","Start of Season Rating","Sire","Dam","Dam's Sire","Same Sire","PP Pre-import races footage","Gear","Comment","B","B_first_time","B_replaced","B_removed","BO","BO_first_time","BO_replaced","BO_removed","CC","CC_first_time","CC_replaced","CC_removed","CP","CP_first_time","CP_replaced","CP_removed","CO",
        "CO_first_time","CO_replaced","CO_removed","E","E_first_time","E_replaced","E_removed","H","H_first_time","H_replaced",
        "H_removed","P","P_first_time","P_replaced","P_removed","PC","PC_first_time","PC_replaced","PC_removed","PS","PS_first_time",
        "PS_replaced","PS_removed","SB","SB_first_time","SB_replaced","SB_removed","SR","SR_first_time","SR_replaced","SR_removed",
        "TT","TT_first_time","TT_replaced","TT_removed","V","V_first_time","V_replaced","V_removed","VO","VO_first_time","VO_replaced",
        "VO_removed","XB","XB_first_time","XB_replaced","XB_removed","HorseCompetitor1","HorseCompetitor2","HorseCompetitor3",
        "HorseCompetitor4","HorseCompetitor5","HorseCompetitor6","HorseCompetitor7","HorseCompetitor8","HorseCompetitor9",
        "HorseCompetitor10","HorseCompetitor11","HorseCompetitor12","HorseCompetitor13","JockeyCompetitor1",
        "JockeyCompetitor2","JockeyCompetitor3","JockeyCompetitor4","JockeyCompetitor5","JockeyCompetitor6",
        "JockeyCompetitor7","JockeyCompetitor8","JockeyCompetitor9","JockeyCompetitor10","JockeyCompetitor11",
        "JockeyCompetitor12","JockeyCompetitor13"
    ]

    # Only keep columns that exist in the DataFrame
    final_cols = [col for col in desired_order if col in race_full_df.columns]
    # Add any remaining columns at the end
    remaining_cols = [col for col in race_full_df.columns if col not in final_cols]
    race_full_df = race_full_df[final_cols + remaining_cols]

    # Output
    mask_2010_2018 = (race_full_df['Date'] >= '2010-01-01') & (race_full_df['Date'] <= '2018-12-31')
    mask_2019_2025 = (race_full_df['Date'] >= '2019-01-01') & (race_full_df['Date'] <= '2025-12-31')
    race_full_df.loc[mask_2010_2018].to_csv("Race_comments_gear_ordered_with_competitors_2010_2018.csv", index=False)
    race_full_df.loc[mask_2019_2025].to_csv("Race_comments_gear_ordered_with_competitors_2019_2025.csv", index=False)

if __name__ == "__main__":
    main()
