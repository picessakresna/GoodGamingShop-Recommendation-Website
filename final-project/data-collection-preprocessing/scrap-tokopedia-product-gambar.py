import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time

# Setup webdriver (Chrome in this example)
driver = webdriver.Chrome()

# Function to download image
def download_image(url, folder_path, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_path, file_name), 'wb') as file:
            file.write(response.content)

df = pd.read_csv('final-project\data-collection-preprocessing\data-produk\clean_product-goodgamingshop.csv')

# Create a new column for the image path
df['image_path'] = ''

# Loop through each product in the df
for index, row in df.iterrows():
    product_id = row['id_produk']
    product_link = row['link']

    # Create a folder for the product
    folder_path = os.path.join('final-project\data-collection-preprocessing\data-produk\images', product_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Use selenium to get the page content
    driver.get(product_link)
    time.sleep(5)  # Adjust this sleep time as necessary to allow the page to fully load

    # Track the number of images downloaded
    images_downloaded = 0

    # Click on the buttons to reveal additional images and download them
    try:
        additional_images_container = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[2]/div[1]/div/div[2]/div/div')
        additional_images = additional_images_container.find_elements(By.TAG_NAME, 'button')

        for i, button in enumerate(additional_images):
            if images_downloaded >= 2:
                break

            # Check if the element with class "playIcon" exists
            try:
                play_icon = button.find_element(By.CLASS_NAME, 'playIcon')

                print(f"Video found for product {product_id}. Skipping this iteration.")
                continue

            except:
                pass

            button.click()
            time.sleep(1)  # Adjust sleep time as necessary for images to load

            # Find the main image using XPath
            try:
                main_image = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[2]/div[2]/div[1]/div/div[1]/button/div/div[2]/img')
                img_url = main_image.get_attribute('src')

                if img_url:
                    # Handle relative URLs
                    if img_url.startswith('/'):
                        img_url = 'https:' + img_url

                    # Download the image
                    image_file_name = f'image_{product_id}_{i}.png'
                    download_image(img_url, folder_path, image_file_name)

                    images_downloaded += 1

                    # # Update the df with the image path
                    # df.at[index, 'image_path'] += os.path.join(folder_path, image_file_name) + '; '

                    # Update the df with the image path
                    df.at[index, 'image_path'] = folder_path

            except Exception as e:
                print(f"Image {i} not found for product {product_id}: {e}")

    except Exception as e:
        print(f"Additional images container not found for product {product_id}: {e}")
        df.at[index, 'image_path'] = ''

# Close the browser
driver.quit()

# Save the updated df to a new CSV file
df.to_csv('updated_products.csv', index=False)

print(df)
