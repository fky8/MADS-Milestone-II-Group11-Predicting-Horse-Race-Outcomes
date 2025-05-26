import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import pandas as pd
import re

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from io import StringIO
import re


OUTPUT_CSV   = "hkjc_horse_profiles.csv"
BASE_DOMAIN  = "https://racing.hkjc.com"

TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[3]'

TABLE_XPATH_ERROR = '//*[@id="innerContent"]/div[2]/div[1]'

# TABLE_XPATH = '//*[@id="innerContent"]/div[2]/div[5]/table'

def get_jockey_name_list():
    """
    Gets a list of jockey names to scrape from the HKJC website.
    Returns a list of jockey names.
    """
    df_record = pd.read_excel('./data/jockey_names.xlsx')
    return df_record['jockey_name'].tolist()

def get_jockey_id_list():
    """
    Gets a list of jockey names to scrape from the HKJC website.
    Returns a list of jockey names.
    """
    df_record = pd.read_excel('./data/jockey_names.xlsx')
    return df_record['jockey_id'].tolist()

SEASONS = [
    "Current", 
    "Previous"
]

BASE_URL = "https://racing.hkjc.com/racing/information/English/Jockey/JockeyPastRec.aspx"


def scrape_all_jockeys():
    """
    Scrape the HKJC 'SelectJockeybyId' index pages for letters A–Z.
    Returns a DataFrame with: Letter, Jockey Name, Rating, URL.
    """

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=options)
    
    df_list = []
    jockey_list = get_jockey_id_list()

    for jockey in jockey_list:
        for season in SEASONS:
            is_last_page = False
            min_race_index = -1
            page_index = 1
            while is_last_page is False:

                # print(f"JockeyId{jockey} season {season} page {page_index}")

                url = f"{BASE_URL}?JockeyId={jockey}&Season={season}&PageNum={page_index}"
                try:
                    driver.get(url)
                except:
                    print(f"    JockeyId{jockey} missing → next jockey")
                    break

                time.sleep(1)
                
                try:
                    tbl = driver.find_element(By.CLASS_NAME, 'ridingRec')
                except:
                    print(f"    JockeyId{jockey} missing → next jockey")
                    break


                try:
                    tbl = driver.find_element(By.CLASS_NAME, 'ridingRec')
                    html = tbl.get_attribute("outerHTML")
                    if 'errorour' in html:
                        print(f"    Jockey#{jockey} No Error")
                        break
                except:
                    continue
                    

                html = tbl.get_attribute("outerHTML")
                tbls = pd.read_html(StringIO(html))
                
                df = tbls[0]
                df['jockey_id'] = jockey
                df['season'] = season

                min_race_index_next = int(df['Race Index'].iloc[-1]) 


                if min_race_index == min_race_index_next:
                    is_last_page = True
                    # print(f"    JockeyId{jockey} last page {page_index}") 
                else:
                    df_list.append(df)
                    min_race_index = min_race_index_next 
                    # print(f"    JockeyId{jockey} page {page_index} min_race_index {min_race_index}")     

                page_index += 1

    if len(df_list) > 0:
        df_concat = pd.concat(df_list, ignore_index=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"jockey_race_data.csv")
        df_concat.to_csv(file_path, index=False)
        print(len(df_concat))
    else:
        print("No data to save.")



    # Build DataFrame
    driver.quit()
    # df = pd.DataFrame(records, columns=["Letter","HorseName","Rating","URL"])
    # print(f"\nDone — total {len(df)} horses scraped")
    # return df



def clean_jockey_data():
    """
    Clean the jockey data DataFrame by removing unnecessary columns
    and renaming columns for consistency.
    """

    df = pd.read_csv('./data/jockey_race_data.csv')
    
    df['race_description'] = df['Race Index'].apply(lambda x: x if '/' in x else None)
    df['Race Index'] = df['Race Index'].apply(lambda x: None if '/' in x else x)
    
    # print(df['race_description'])


    race_description = df['race_description'].to_list()

    # for 
    race_desc_list = []
    race_sub_index = []
    j = 0
    for i, val in df['race_description'].items():
        j += 1
        if pd.isna(val):
            race_desc_list.append(current_race_description)
            race_sub_index.append(j)
        else:
            current_race_description = val
            race_desc_list.append(current_race_description)
            race_sub_index.append(j)
            j = 0
    
    df['Race Sub Index'] = race_sub_index
    df['race_description'] = pd.Series(race_desc_list) 
    df = df[df['Race Index'].notna()]
    df['race_date'] = df['race_description'].apply(lambda x: pd.to_datetime(x[0:10], format='%d/%m/%Y', errors='coerce'))
    df['race_course'] = df['race_description'].apply(lambda x: 'Happy Valley Jockey Challenge' if 'Happy Valley Jockey Challenge' in x else 'Sha Tin Jockey Jockey Challenge')
    df['race_wins'] = df['race_description'].apply(lambda x: x[x.index('('):x.index(')')] if '(' in x and ')' in x else None)
    df[['first_place','first_place_win', 'second_place', 'second_place_win', 'third_place', 'third_place_win']] = df['race_wins'].str.split(' ', expand=True)
    df.drop(['first_place', 'second_place', 'third_place'], axis=1, inplace=True)

    df['first_place_win'] = df.apply(lambda x: int(x['first_place_win'].replace('*', '')) if x['first_place_win'] is not None else 0  if x['Race Sub Index'] == 1 else 0, axis=1) 
    df['second_place_win'] = df.apply(lambda x: int(x['second_place_win'].replace('*', '')) if x['second_place_win'] is not None else 0 if x['Race Sub Index'] == 1 else 0, axis=1 )
    df['third_place_win'] = df.apply(lambda x: int(x['third_place_win'].replace('*', '')) if x['third_place_win'] is not None else 0 if x['Race Sub Index'] == 1 else 0, axis=1 )
    
    # df_group = df[['race_date', 'jockey_id', 'season', 'first_place_win', 'second_place_win', 'third_place_win']].groupby(['race_date', 'jockey_id', 'season'], sort=True).agg({
    #     'first_place_win': 'sum', 'second_place_win': 'sum', 'third_place_win': 'sum'        
    # }).reset_index()
    # print(df_group.tail(10))


    new = df[['race_date', 'jockey_id', 'season', 'first_place_win', 'second_place_win', 'third_place_win']].groupby(['race_date', 'jockey_id', 'season']).apply(lambda x: x.iloc[::-1].cumsum()).reset_index()
    print(new[new['jockey_id']=='PZ'].tail(10))

    # df['cumulative_second_place_win'] =  df.groupby(['jockey_id', 'season'])['second_place_win'].apply(lambda x: x.iloc[::-1].cumsum().iloc[::-1])
    # df['cumulative_second_place_win'] =  df.groupby(['jockey_id', 'season'])['second_place_win'].apply(lambda x: x.iloc[::-1].cumsum().iloc[::-1])
    # df['cumulative_third_place_win'] =  df.groupby(['jockey_id', 'season'])['third_place_win'].apply(lambda x: x.iloc[::-1].cumsum().iloc[::-1])     


    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, f"data/jockey_race_data_clean.csv")
    df.to_csv(file_path, index=False)


    return df

def main(df_idx):
    print('main')

if __name__ == "__main__":
    # df = scrape_all_horses()
    # main(df)
    # scrape_all_jockeys()
    clean_jockey_data()


# %%
