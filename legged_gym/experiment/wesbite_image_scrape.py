from legged_gym import LEGGED_GYM_ROOT_DIR
import hashlib, io, requests
from selenium import webdriver
from selenium.webdriver import ChromeOptions
from bs4 import BeautifulSoup
from pathlib import Path
from PIL import Image
import threading

base_url = "https://www.manytextures.com/"
def gets_url():
    results = []
    for item in soup.find_all('img'): 
        # filter out non-jpg
        if item['src'].endswith('.jpg'):
            print(base_url+item['src'])
            results.append(base_url+item['src'])
    return results

def scrape_page(i, options, returned_results):
    print(f'Getting page {i}')
    driver = webdriver.Chrome(options=options)
    driver.get(f"https://www.manytextures.com/?p={i}")
    content = driver.page_source
    soup = BeautifulSoup(content, "html.parser")
    driver.quit()
    for item in soup.find_all('img'): 
        # filter out non-jpg
        if item['src'].endswith('.jpg'):
            print(item['src'])
            returned_results.append(item['src'])

if __name__ == "__main__":
    options = ChromeOptions()
    options.add_argument("--headless=new")

    returned_results = []

    # Assuming options and returned_results are defined
    # threads = []
    # for i in range(1, 501):
    #     t = threading.Thread(target=scrape_page, args=(i, options, returned_results))
    #     t.start()
    #     threads.append(t)

    # # Wait for all threads to finish
    # for t in threads:
    #     t.join()
    
    for i in range(1, 31):
        print(f'Getting page {i}')
        driver = webdriver.Chrome(options=options)
        curr_url = f"https://www.manytextures.com/most_popular/?p={i}"
        driver.get(curr_url)
        content = driver.page_source
        soup = BeautifulSoup(content, "html.parser")
        driver.quit()
        returned_results += gets_url()

    print(f'Found {len(returned_results)} images')
    for b in returned_results:
        file_path = Path(LEGGED_GYM_ROOT_DIR+'/resources/textures/', b.split('/')[-1])
        if file_path.exists():
            continue
        image_content = requests.get(b).content
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert("RGB")
        image.save(file_path, "JPEG", quality=80)