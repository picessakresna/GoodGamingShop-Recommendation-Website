import csv
import os
import pandas as pd
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.alert import Alert
from time import sleep

driver = webdriver.Chrome()

def loaded_page(self, element):
    global myElem
    delay = 5
    try:
        myElem = WebDriverWait(self, delay).until(EC.presence_of_element_located((By.XPATH, element)))
    except TimeoutException:
        print('Loading too much time')

    return myElem

def save_reviewed_product_to_csv(directory, nama_produk):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, 'product-reviewed.csv')
    
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Nama Produk']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nama Produk']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'Nama Produk': nama_produk})

def save_reviews_to_csv(directory, nama_produk_judul, nama_produk, reviews, file_number):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_path = os.path.join(directory, f'{file_number}_{nama_produk_judul}_reviews.csv')
    if os.path.exists(file_path):
        print(f"Reviews file for '{nama_produk_judul}' already exists. Skipping saving.")
        return
    
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nama Produk', 'Review', 'Nama Akun', 'Rating User', 'Ulasan Produk', 'Waktu Review', 'Varian Produk']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for review in reviews:
            review['Nama Produk'] = nama_produk
            writer.writerow(review)

def get_ulasan():
    i = 1
    review_count = 0
    count_review_element = driver.find_element(By.XPATH, '//*[@data-testid="reviewSortingSubtitle"]')
    count_review_text = count_review_element.text
    count = int(count_review_text.split(" ")[-2])
    review_data = []

    while i <= 50:
        try:
            nama_akun = loaded_page(driver, f'//*[@id="review-feed"]/article[{i}]/div/div[2]/span').text

            rating_user_element = driver.find_element(By.XPATH, f'//*[@id="review-feed"]/article[{i}]/div/div[1]/div/div')
            rating_user = rating_user_element.get_attribute('aria-label')

            waktu_review = loaded_page(driver, f'//*[@id="review-feed"]/article[{i}]/div/div[1]/div/p').text

            xpath_article = f'//*[@id="review-feed"]/article[{i}]'

            try:
                varian_element = driver.find_element(By.XPATH, xpath_article + '//*[@data-testid="lblVarian"]')
                varian_product = varian_element.text
            except NoSuchElementException:
                varian_product = "none"

            try:
                ulasan_produk_element = driver.find_element(By.XPATH, f'{xpath_article}//*[@data-testid="lblItemUlasan"]')
                ulasan_produk = ulasan_produk_element.text
            except NoSuchElementException:
                ulasan_produk = "none"

        except NoSuchElementException:
            pass

        review_count += 1
        print(f'Review {review_count}: {nama_akun}, {rating_user}, {ulasan_produk}')
        
        review_data.append({
            'Review': review_count,
            'Nama Akun': nama_akun,
            'Rating User': rating_user,
            'Ulasan Produk': ulasan_produk,
            'Waktu Review': waktu_review,
            'Varian Produk': varian_product
        })
        
        if review_count == count:
            break

        i += 1
        
        if i > 50:
            button_next = loaded_page(driver, f'//*[@id="zeus-root"]/div/main/div[2]/div[1]/div[2]/section/div[3]/nav/ul/li[11]/button')
            button_next.click()
            sleep(1)
            i = 1

    return review_data

def load_ulasan(link):
    driver.get(link)
    sleep(1)
    try:
        # Scroll to the review section
        review_section = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_detail_media"]')
        driver.execute_script("arguments[0].scrollIntoView();", review_section)
        sleep(1)
        
        # Check if review section is empty
        try:
            empty_state_element = driver.find_element(By.XPATH, '//*[@id="review-feed"]/div/div[@data-unify="EmptyState"]')
            print("No reviews found.")
            return None
        except NoSuchElementException:
            pass

        # Click the "View All Feedback" button
        all_review_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//*[@data-testid="btnViewAllFeedback"]')))
        href_value = all_review_button.get_attribute("href")
        driver.get(href_value)
        sleep(1)

        # Get reviews
        reviews = get_ulasan()
    except NoSuchElementException:
        reviews = None

    return reviews

def get_link():
    df = pd.read_csv('product-scrap-data/data-produk/product-goodgamingshop.csv')

    file_number = 1  # Initialize file number

    for index, row in df.iloc[0:].iterrows():
        nama_produk_judul = ' '.join(row['Nama Produk'].split()[:1])  # Limit to first four words
        nama_produk = row['Nama Produk']
        link = row['Link']
        
        reviews = load_ulasan(link)
        
        if reviews is not None:
            save_reviews_to_csv('product-scrap-data/data-ulasan', nama_produk_judul, nama_produk, reviews, file_number)
            save_reviewed_product_to_csv('product-scrap-data/data-produk', nama_produk)
            file_number += 1  # Increment file number for the next iteration
        else:
            print(f"No reviews found for '{nama_produk_judul}'. CSV file not saved.")

def main():
    get_link()
    
if __name__ == "__main__":
    main()