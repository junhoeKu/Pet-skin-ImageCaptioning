from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains as ac
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

import requests
from bs4 import BeautifulSoup
import pandas as pd
import csv
import os
import time
from datetime import datetime, timedelta
from random import *

def initialize_driver():
    chrome_options = Options()
    chrome_options.add_argument('headless')  # 브라우저 창을 띄우지 않고 실행
    chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])
    service = Service(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def read_urls_from_file(file_path):
    with open(file_path, 'r') as f:
        page_urls = f.readlines()
    return page_urls

def crawl_expert_data(driver, page_urls, output_file_path):
    try:
        with open(output_file_path, "a", encoding='utf-8') as f:
            fieldnames = ['idx', 'title', 'question', 'answer', 'images', 'url']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            start_idx = 0
            for url in page_urls[0:-1]:
                driver.get(url.strip())
                
                # Alert 처리
                try:
                    result = driver.switch_to.alert
                    print(f"{start_idx}: result", result)
                    result.accept()
                    data = {'idx': start_idx, 'title': '@Exception_alert@', 'question': "", 'answer': "", 'images': 'none', 'url': url}
                    writer.writerow(data)
                    start_idx += 1
                    continue
                except Exception as e:
                    pass
                
                # 질문 타이틀 크롤링
                try:
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'question-content')))
                    pros = driver.find_element(By.CLASS_NAME, 'question-content')
                    title = pros.find_element(By.CLASS_NAME, 'title').text
                    print(f"{start_idx}: title", title)
                except Exception as e:
                    title = "@Exception_title@"
                    print(f"{start_idx}: title", title)
                    print('title: 예외가 발생했습니다.', e)  
                    
                # 질문 텍스트 크롤링
                try:
                    tags = pros.find_element(By.CLASS_NAME, 'c-heading__content')
                    question_txt = tags.text
                except Exception as e:
                    question_txt = ""
                    print('question_txt: 예외가 발생했습니다.', e)
                    
                # 이미지 URL 크롤링
                try:
                    images = pros.find_elements(By.TAG_NAME, 'img')
                    image_urls = [img.get_attribute('src') for img in images]
                    if not image_urls:
                        image_urls = ['none']
                except Exception as e:
                    image_urls = ['none']
                    print('images: 예외가 발생했습니다.', e)
                    
                # 답변 크롤링
                answer_list = driver.find_elements(By.CLASS_NAME, "answer-content__item")
                t = ""
                new_answer_list = []
                try:
                    for n, answer in enumerate(answer_list):
                        profile = answer.find_element(By.CLASS_NAME, 'profile_card').text
                        if profile.find('수의사') == -1:
                            continue
                        new_answer_list.append(answer.find_element(By.CLASS_NAME, '_endContentsText').text)
                    
                    if len(new_answer_list) >= 2:
                        print('수의사 답변이 2개 이상입니다.')
                        t = "@Excepiton 수의사 답변이 2개 이상입니다."
                    else:
                        t = new_answer_list[0] if new_answer_list else ""
                        
                except Exception as e:
                    t = "@Excepiton"
                    print('answer: 예외가 발생했습니다.', e)   
                
                # CSV에 저장
                data = {
                    'idx': start_idx, 
                    'title': title, 
                    'question': question_txt, 
                    'answer': t, 
                    'images': ' | '.join(image_urls), 
                    'url': url
                }
                writer.writerow(data) 
                
                start_idx += 1
                
    except Exception as e:
        print('###### 예외가 발생했습니다', e)
    finally:
        driver.quit()

def main():
    # 설정된 파일 경로
    url_file_path = "./result/alert_url_2.txt"
    output_file_path = "C:/VScode/mutimodal/expert_list.csv"

    # 드라이버 초기화
    driver = initialize_driver()
    
    # URL 목록 읽기
    page_urls = read_urls_from_file(url_file_path)
    
    # 데이터 크롤링
    crawl_expert_data(driver, page_urls, output_file_path)

if __name__ == "__main__":
    main()
