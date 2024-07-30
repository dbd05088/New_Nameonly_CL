import os
import logging
import requests
import time
from typing import List
from .base_generator import URLGenerator
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

class GoogleURLGenerator(URLGenerator):
    def __init__(self, save_dir, mode='headless', use_color=False, use_size=False, max_scroll=10, sleep_time=3):
        super().__init__()
        self.save_dir = save_dir
        self.options = Options()
        self.use_color = use_color
        self.use_size = use_size
        self.max_scroll = max_scroll
        self.sleep_time = sleep_time
        self.url_list = None
        if mode == 'headless':
            self.options.add_argument("--headless")
            self.options.add_argument("--no-sandbox")
            self.options.add_argument("--disable-dev-shm-usage")
            self.options.add_argument("--disable-gpu")
        
        self.colors_list = ['red', 'orange', 'yellow', 'green', 'teal', 'blue', 'purple', 'pink', 'white', 'gray', 'black', 'brown']        
        self.size_list = ['l', 'm', 'i']
        # self.driver = webdriver.Chrome(options=self.options)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
    
    def scroll_down(self):
        num_scroll = 0
        while num_scroll < self.max_scroll:
            # time.sleep(self.sleep_time)
            # Scroll down to bottom
            self.driver.find_element(By.XPATH, '//body').send_keys(Keys.END)
            num_scroll += 1
            logger.info('Scroll down to bottom')
            time.sleep(self.sleep_time)
            try:
                # Click on the 'Load more' button
                load_more_button = self.driver.find_element(By.XPATH, '//*[@id="islmp"]/div/div/div/div/div[1]/div[2]/div[2]/input')
                if load_more_button.is_displayed():
                    load_more_button.click()
            except:
                logger.info('No more load more button')
            time.sleep(self.sleep_time)
            try:
                # Check if there is no more content to load
                no_more_content = self.driver.find_element(By.XPATH, '//div[@class="K25wae"]//*[text()="Looks like you\'ve reached the end"]')
                if no_more_content.is_displayed():
                    break
            except:
                pass
    
    def crawl_color_size(self, query: str, color=None, size=None):
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.maximize_window()
        if color is None:
            if size is None:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&hl=en"
            else:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=isz:{size}&hl=en"
        else:
            if size is None:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=ic:specific%2Cisc:{color}&hl=en"
            else:
                URL = f"https://www.google.com/search?q={query}&tbm=isch&tbs=ic:specific%2Cisc:{color}%2Cisz:{size}&hl=en"

        logger.info(f"URL: {URL}")
        self.driver.get(URL)
        try:
            self.scroll_down()
        except:
            logger.info('Error in scrolling down')
        
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        image_info_list = soup.find_all('img', class_='YQ4gaf')

        # Google image search result contains two types of image URLs: 'data-src' and 'src'
        # 1. 'src' attribute
        # 2. 'data-src' attribute
        for i in range(len(image_info_list)):
            if 'src' in image_info_list[i].attrs:
                self.url_list.append(image_info_list[i]['src'])
            elif 'data-src' in image_info_list[i].attrs:
                self.url_list.append(image_info_list[i]['data-src'])
        
        logger.info(f'Number of images until size {size}: {len(self.url_list)}')
        
        self.driver.quit()
    
    
    def generate_url(self, query: str, total_images: int = 10000, image_type=None, filename=None):
        if filename is None:
            filename = query
        self.url_list = []
        if self.use_color:
            for color in self.colors_list:
                if self.use_size:
                    for size in self.size_list:
                        self.crawl_color_size(query.replace('_', ' '), color=color, size=size)
                        if len(set(self.url_list)) >= total_images:
                            logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                            break
                else:
                    self.crawl_color_size(query.replace('_', ' '), color=color, size=None)
        else:
            if self.use_size:
                for size in self.size_list:
                    self.crawl_color_size(query.replace('_', ' '), color=None, size=size)
                    if len(set(self.url_list)) >= total_images:
                        logger.info(f"Break the loop because the number of images is enough: {len(set(self.url_list))}")
                        break
            else:
                self.crawl_color_size(query.replace('_', ' '), color=None, size=None)
        
        # Save the image URLs
        logger.info(f"Total number of images (after removing duplicated urls): {len(set(self.url_list))}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.url_list = list(set(self.url_list))
    
        # Remove urls not starting with 'http'
        self.url_list = [url for url in self.url_list if url.startswith('http')]
        logger.info(f"Total number of images (after removing urls not starting with 'http'): {len(self.url_list)}")
        
        self.driver.quit()
        return self.url_list
    
        # with open(os.path.join(self.save_dir, f'{filename}.txt'), 'w') as f:
        #     for url in self.url_list:
        #         f.write(url + '\n')
