import os
import time
import random
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
import undetected_chromedriver as uc

def get_soup(driver, url):
    driver.get(url)
    time.sleep(random.uniform(2, 4))
    return BeautifulSoup(driver.page_source, 'html.parser')

def extract_details(soup):
    details = {
        'purpose': None,
        'bedrooms': None,
        'bathrooms': None,
        'added': None,
        'description': None,
        'schools_nearby': None,
        'restaurants_nearby': None,
        'hospitals_nearby': None,
        'parks_nearby': None,
        'price_index_now': None,
        'price_index_6mo': None,
        'price_index_12mo': None,
        'price_index_24mo': None,
        'price_index_segment': None,
        'main_features': [],
        'room_features': []
    }

    # --- DETAILS BLOCK ---
    try:
        for li in soup.select('ul[aria-label="Property details"] li'):
            label = li.select_one('span[aria-label]')
            if not label:
                continue
            key = label['aria-label'].lower()
            val = label.get_text(strip=True)
            if 'purpose' in key:
                details['purpose'] = val
            elif 'beds' in key:
                details['bedrooms'] = val
            elif 'baths' in key:
                details['bathrooms'] = val
            elif 'creation date' in key or 'added' in key:
                details['added'] = val
    except:
        pass

    # --- DESCRIPTION ---
    try:
        desc = soup.select_one('div[aria-label="Property description"] span')
        if desc:
            details['description'] = desc.get_text(strip=True).replace('\n', ' ')
    except:
        pass

    # --- NEARBY PLACES ---
    for place_type in ['Schools', 'Restaurants', 'Hospitals', 'Parks']:
        try:
            block = soup.find('div', class_='_45812d93', string=place_type)
            parent = block.find_parent('div', class_='ff476d2b') if block else None
            text = parent.select_one('.c6add38a').text.strip() if parent else None
            details[f"{place_type.lower()}_nearby"] = text
        except:
            pass

    # --- PRICE INDEX ---
    try:
        segment = soup.select_one("div._83bb17d1 > h2")
        details['price_index_segment'] = segment.text.strip() if segment else None

        # Current price index
        price_items = soup.select("ul._2ac2a083 li._048ce3b0")
        for item in price_items:
            label = item.select_one('div._7e229aa3')
            value = item.select_one('div._88ac7791')
            if not label:
                continue
            if 'Current' in label.text and value:
                details['price_index_now'] = value.text.strip()

        # Historical price changes
        ul_price_changes = soup.find('ul', class_='ea2b146e')
        if ul_price_changes:
            for li in ul_price_changes.find_all('li'):
                spans = li.find_all('span')
                if len(spans) == 2:
                    label = spans[0].text.strip().lower()
                    val = spans[1].text.strip()
                    if '6 months' in label:
                        details['price_index_6mo'] = val
                    elif '12 months' in label:
                        details['price_index_12mo'] = val
                    elif '24 months' in label:
                        details['price_index_24mo'] = val
    except:
        pass

    # --- FEATURES ---
    for feature_group in [('Main Features', 'main_features'), ('Rooms', 'room_features')]:
        try:
            heading = soup.find('div', class_='d0142259', string=feature_group[0])
            if heading:
                ul = heading.find_next('ul')
                items = [li.text.strip() for li in ul.select('li span._9121cbf9')]
                details[feature_group[1]] = items
        except:
            pass

    return details

# --- SCRAPE ALL LISTING FILES ---
options = uc.ChromeOptions()
driver = uc.Chrome(options=options)

listing_folder = 'zameen.com/listings'
output_folder = 'zameen.com/processed_listings'
os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist

# Define all possible columns from extract_details
detail_columns = [
    'purpose', 'bedrooms', 'bathrooms', 'added', 'description',
    'schools_nearby', 'restaurants_nearby', 'hospitals_nearby', 'parks_nearby',
    'price_index_now', 'price_index_6mo', 'price_index_12mo', 'price_index_24mo',
    'price_index_segment', 'main_features', 'room_features'
]

for file in os.listdir(listing_folder):
    if not file.endswith('.csv'):
        continue

    filepath = os.path.join(listing_folder, file)
    df = pd.read_csv(filepath)

    # Initialize column order: input columns + detail columns before 'link'
    output_cols = list(df.columns)
    insert_at = output_cols.index('link') if 'link' in output_cols else len(output_cols)
    output_cols = output_cols[:insert_at] + detail_columns + ['link', 'scraped_at']

    output_filepath = os.path.join(output_folder, f"processed_{file}")

    # Initialize the output CSV with all headers
    if not os.path.exists(output_filepath):
        pd.DataFrame(columns=output_cols).to_csv(output_filepath, index=False)

    print(f"📄 Processing file: {file} (total rows: {len(df)})")

    for index, row in df.iterrows():
        try:
            soup = get_soup(driver, row['link'])
            detail_data = extract_details(soup)

            # Preserve original 'type' if not overridden
            if 'type' in row and 'type' in detail_data:
                new_type = detail_data.get('type')
                old_type = row.get('type')
                if new_type and new_type.strip() not in ['-', '', None] and new_type != old_type:
                    detail_data['type'] = new_type
                else:
                    detail_data['type'] = old_type

            # Combine original row data with scraped details
            combined = {**row.to_dict(), **detail_data}
            combined['scraped_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Create a single-row DataFrame
            row_df = pd.DataFrame([combined])

            # Ensure the row DataFrame has the correct column order
            row_df = row_df[output_cols]

            # Append the row to the single output CSV
            row_df.to_csv(output_filepath, mode='a', header=False, index=False)

            print(f"✅ Scraped and appended: {row['link']} to {output_filepath}")

        except Exception as e:
            print(f"⚠ Failed: {row['link']} → {e}")

driver.quit()