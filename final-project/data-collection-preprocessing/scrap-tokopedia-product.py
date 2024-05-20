import csv
import os
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
scrapecounter = 0  # Initialize scrape counter

def loaded_page(self, element):
    global myElem
    delay = 5
    try:
        myElem = WebDriverWait(self, delay).until(EC.presence_of_element_located((By.XPATH, element)))
    except TimeoutException:
        print('Loading too much time')

    return myElem

def save_produk_info_to_csv(product_info):
    directory = 'data-collection-preprocessing/data-produk'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define file path
    file_path = os.path.join(directory, 'product-goodgamingshop.csv')

    # Check if file exists, if not, create and write header
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Nama Produk', 'Kategori', 'Jumlah', 'Harga Jual', 'Harga Awal', 'Diskon', 'Deskripsi', 'Rating', 'Rating Counter', 'Link']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Append product info to CSV file
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Nama Produk', 'Kategori', 'Jumlah', 'Harga Jual', 'Harga Awal', 'Diskon', 'Deskripsi', 'Rating', 'Rating Counter', 'Link']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        for info in product_info:
            writer.writerow(info)


def get_produkinfo():
    global scrapecounter  # Access the global scrape counter variable

    nama_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/h1').text
    scrapecounter += 1  # Increment scrape counter

    kategori1_produk = loaded_page(driver, '//*[@id="main-pdp-container"]/div[1]/nav/ol/li[1]').text
    kategori2_produk = loaded_page(driver, '//*[@id="main-pdp-container"]/div[1]/nav/ol/li[2]').text
    kategori3_produk = loaded_page(driver, '//*[@id="main-pdp-container"]/div[1]/nav/ol/li[3]').text
    kategori4_produk = loaded_page(driver, '//*[@id="main-pdp-container"]/div[1]/nav/ol/li[4]').text
    kategori_produk = '|'.join([kategori1_produk, kategori2_produk, kategori3_produk, kategori4_produk])

    try:
        jumlah_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[1]')
        jumlah_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[1]').text
    except NoSuchElementException:
        try:
            jumlah_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p')
            jumlah_produk = jumlah_produk_element.text
        except NoSuchElementException:
            jumlah_produk = 0

    try:
        hargajual_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[2]/div[2]')
        hargajual_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div[1]').text
    except NoSuchElementException:
        hargajual_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div').text

    try:
        hargaawal_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[2]/div[2]')
        hargaawal_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div[2]/div[2]/span[2]').text
    except NoSuchElementException:
        try:
            hargaawal_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p')
            hargaawal_produk = hargajual_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div[1]').text
        except NoSuchElementException:
            hargaawal_produk = hargajual_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div').text

    try:
        diskon_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[2]/div[2]')
        diskon_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[2]/div[2]/div[1]/span[2]').text
    except NoSuchElementException:
        try:
            diskon_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p')
            diskon_produk = 0
        except NoSuchElementException:
            diskon_produk = 0

    deskripsi_produk = loaded_page(driver, '//*[@id="pdp_comp-product_detail"]/div[2]/div[2]/div/span/span/div').text

    try:
        rating_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[1]')
        rating_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[2]/span[1]/span[2]').text
    except NoSuchElementException:
        try:
            rating_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p')
            rating_produk = 0
        except NoSuchElementException:
            try:
                rating_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[2]')
                rating_produk = 0
            except NoSuchElementException:
                rating_produk = 0

    try:
        ratingcounter_produk_element = driver.find_element(By.XPATH,'//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[1]')
        ratingcounter_produk = loaded_page(driver, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p[2]/span[2]').text
    except NoSuchElementException:
        try:
            ratingcounter_produk_element = driver.find_element(By.XPATH, '//*[@id="pdp_comp-product_content"]/div/div[1]/div/p')
            ratingcounter_produk = 0
        except NoSuchElementException:
            ratingcounter_produk = 0

    link_produk = driver.current_url

    # Append product info to CSV
    product_info = [{
        'Nama Produk': nama_produk,
        'Kategori': kategori_produk,
        'Jumlah': jumlah_produk,
        'Harga Jual': hargajual_produk,
        'Harga Awal': hargaawal_produk,
        'Diskon': diskon_produk,
        'Deskripsi': deskripsi_produk,
        'Rating': rating_produk,
        'Rating Counter': ratingcounter_produk,
        'Link': link_produk
    }]

    save_produk_info_to_csv(product_info)
    kata = nama_produk.split()
    info_nama = kata[:3]
    print_produk = ' '.join(info_nama)
    print(f'Jumlah Produk Terscrape = {scrapecounter}, add {print_produk} reviews to CSV')

def get_allproduct():
    for i in range(1, 80):
        try:
            alamatbarang = loaded_page(driver, f'//*[@id="zeus-root"]/div/div[2]/div[2]/div[4]/div/div[2]/div[1]/div[{i}]/div/div/div/div/div/div[1]/a').get_attribute('href')
            driver.get(alamatbarang)
            sleep(1)

            get_produkinfo()

            driver.back()
            sleep(1)
        except NoSuchElementException:
            print('No more product')
            return False
    return True

def get_merchantinfo():
    i = 1
    while True:
        try:
            driver.get(f"https://www.tokopedia.com/goodgamingshop/product/page/{i}?sort=8")
            sleep(2)
            if not get_allproduct():
                print('Scrapping selesai')
                break
            i += 1
        except TimeoutException:
            print('Loading too much time')
            break

def main():
    get_merchantinfo()
    
if __name__ == "__main__":
    main()
