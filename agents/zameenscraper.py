# scrape_listing_links.py

import os
from datetime import datetime, timezone
import time
import re
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import undetected_chromedriver as uc

def parse_price(price_str):
    price_str = price_str.replace("PKR", "").replace(",", "").strip().lower()
    if "crore" in price_str:
        return float(re.findall(r"\d+\.?\d*", price_str)[0]) * 1e7
    elif "lakh" in price_str:
        return float(re.findall(r"\d+\.?\d*", price_str)[0]) * 1e5
    else:
        return float(re.findall(r"\d+\.?\d*", price_str)[0])

def parse_area(area_str):
    if "kanal" in area_str.lower():
        value = float(re.findall(r"\d+\.?\d*", area_str)[0])
        return value * 20
    elif "marla" in area_str.lower():
        return float(re.findall(r"\d+\.?\d*", area_str)[0])
    return None

# Create folders if not exist
os.makedirs("zameen.com/listings", exist_ok=True)

cities = {
    #"Rawalpindi_Homes": "https://www.zameen.com/Homes/Rawalpindi-41-1.html",
    #"Rawalpindi_Plots": "https://www.zameen.com/Plots/Rawalpindi-41-1.html",
    #"Rawalpindi_Commercial": "https://www.zameen.com/Commercial/Rawalpindi-41-1.html",
    #"Islamabad_Homes": "https://www.zameen.com/Homes/Islamabad-3-1.html",
    #"Islamabad_Plots": "https://www.zameen.com/Plots/Islamabad-3-1.html"
    #"Islamabad_Commercial": "https://www.zameen.com/Commercial/Islamabad-3-1.html",
    #"Lahore_Homes": "https://www.zameen.com/Homes/Lahore-1-1.html",
    #"Lahore_Plots": "https://www.zameen.com/Plots/Lahore-1-1.html",
    #"Lahore_Commercial": "https://www.zameen.com/Commercial/Lahore-1-1.html",
    #"Karachi_Homes": "https://www.zameen.com/Homes/Karachi-2-1.html",
    #"Karachi_Plots": "https://www.zameen.com/Plots/Karachi-2-1.html",
    #"Karachi_Commercial": "https://www.zameen.com/Commercial/Karachi-2-1.html",
    #"Faisalabad_Homes": "https://www.zameen.com/Homes/Faisalabad-4-1.html",
    #"Faisalabad_Plots": "https://www.zameen.com/Plots/Faisalabad-4-1.html",
    #"Faisalabad_Commercial": "https://www.zameen.com/Commercial/Faisalabad-4-1.html",
    #"Multan_Commercial": "https://www.zameen.com/Commercial/Multan-5-1.html",
    #"Multan_Homes": "https://www.zameen.com/Homes/Multan-5-1.html",
    #"Multan_Plots": "https://www.zameen.com/Plots/Multan-5-1.html",
    #"Gujranwala_Commercial": "https://www.zameen.com/Commercial/Gujranwala-6-1.html",
    #"Gujranwala_Homes": "https://www.zameen.com/Homes/Gujranwala-6-1.html",
    #"Gujranwala_Plots": "https://www.zameen.com/Plots/Gujranwala-6-1.html",
    #"Sialkot_Commercial": "https://www.zameen.com/Commercial/Sialkot-7-1.html",
    #"Sialkot_Homes": "https://www.zameen.com/Homes/Sialkot-7-1.html",
    #"Sialkot_Plots": "https://www.zameen.com/Plots/Sialkot-7-1.html",
    #"Peshawar_Commercial": "https://www.zameen.com/Commercial/Peshawar-8-1.html",
    #"Peshawar_Homes": "https://www.zameen.com/Homes/Peshawar-8-1.html",
    #"Peshawar_Plots": "https://www.zameen.com/Plots/Peshawar-8-1.html",
    #"Quetta_Commercial": "https://www.zameen.com/Commercial/Quetta-9-1.html",
    #"Quetta_Homes": "https://www.zameen.com/Homes/Quetta-9-1.html",
    #"Quetta_Plots": "https://www.zameen.com/Plots/Quetta-9-1.html"
}

options = uc.ChromeOptions()
driver = uc.Chrome(options=options)

for city, city_url in cities.items():
    results = []
    driver.get(city_url)
    time.sleep(5)
    print(f"\n📍 City: {city}")

    try:
        location_elems = WebDriverWait(driver, 15).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'ul._10e87735 li._6d61972c a'))
        )
        location_links = [a.get_attribute("href") for a in location_elems]
    except:
        print("❌ Failed to get locations for city:", city)
        continue

    for loc_url in location_links:
        print(f"➡️ Location URL: {loc_url}")
        page = 1
        while True:
            paged_url = re.sub(r"-\d+\.html$", f"-{page}.html", loc_url)
            print(f"🌐 Visiting: {paged_url}")
            driver.get(paged_url)
            time.sleep(4)

            listings = driver.find_elements(By.CSS_SELECTOR, "li[role='article']")
            if not listings:
                print("🛑 No more listings. Moving to next location.")
                break

            for listing in listings:
                try:
                    title_elem = listing.find_element(By.CSS_SELECTOR, "h2[aria-label='Title']")
                    price_elem = listing.find_element(By.CSS_SELECTOR, "h4 span[aria-label='Price']")
                    location_elem = listing.find_element(By.CSS_SELECTOR, "div[aria-label='Location']")
                    area_elem = listing.find_element(By.CSS_SELECTOR, "span[aria-label='Area']")
                    link = listing.find_element(By.TAG_NAME, 'a').get_attribute('href')

                    title = title_elem.text.strip()
                    location = location_elem.text.strip()
                    price = parse_price(price_elem.text)
                    area = parse_area(area_elem.text)
                    price_per_marla = round(price / area, 2) if area else None

                    if any(x in title.lower() for x in ["flat", "apartment", "unit", "suite"]):
                        ptype = "Flat"
                    elif any(x in title.lower() for x in ["house", "home"]):
                        ptype = "House"
                    elif "plot" in title.lower():
                        ptype = "Plot"
                    elif any(x in title.lower() for x in ["shop", "store"]):
                        ptype = "Shop"
                    elif "office" in title.lower():
                        ptype = "Office"
                    elif any(x in title.lower() for x in ["farmhouse", "farm house"]):
                        ptype = "Farmhouse"
                    elif "building" in title.lower():
                        ptype = "Building"
                    elif "factory" in title.lower():
                        ptype = "Factory"
                    elif any(x in title.lower() for x in ["warehouse", "ware house"]):
                        ptype = "Warehouse"
                    else:
                        ptype = "Other"

                    results.append({
                        "title": title,
                        "city": city,
                        "location": location,
                        "price": price,
                        "area_marla": area,
                        "price_per_marla": price_per_marla,
                        "type": ptype,
                        "scraped_at": datetime.now(timezone.utc).isoformat(),
                        "link": link
                    })

                except Exception as e:
                    print("⚠️ Skipping listing due to error:", e)

            page += 1

    df = pd.DataFrame(results)
    df.to_csv(f"zameen.com/listings/{city}.csv", index=False)
    print(f"✅ Saved: zameen.com/listings/{city}.csv")

driver.quit()
