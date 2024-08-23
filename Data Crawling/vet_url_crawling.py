
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService
from bs4 import BeautifulSoup
from datetime import datetime
import time
import openpyxl
from openpyxl.styles import PatternFill, Color
from openpyxl import Workbook
from random import uniform

def initialize_driver(geckodriver_path, useragent, proxy_host="127.0.0.1", proxy_port=9050):
    # Firefox profile settings
    profile = webdriver.FirefoxProfile()
    profile.set_preference('general.useragent.override', useragent)
    profile.set_preference("network.proxy.type", 1)
    profile.set_preference("network.proxy.socks", proxy_host)
    profile.set_preference("network.proxy.socks_port", proxy_port)

    # Firefox options
    options = FirefoxOptions()
    options.profile = profile

    # Specify the path to the geckodriver executable
    service = FirefoxService(executable_path=geckodriver_path)

    # Initialize the WebDriver
    driver = webdriver.Firefox(service=service, options=options)
    return driver

def sort_kind(index):
    # 추천순
    if index == 1:
        return 'vcount'
    # 최신순
    elif index == 2:
        return 'date'
    # 정확도순
    else:
        return 'none'

def scrape_expert_responses(driver, expert_list_path, output_path, sort_order=2, start_date='2018.11.01', end_date='2024.06.24', max_pages=500):
    _sort_kind = sort_kind(sort_order)
    date = str(datetime.now()).replace('.', '_').replace(' ', '_')
    
    # 전문가 리스트 URL 읽기
    with open(expert_list_path, 'r') as f:
        expert_page = f.readlines()
    
    # 각 전문가 답변 URL 저장할 파일 열기
    output_file = f"{output_path}/expert_list_{date}.txt"
    with open(output_file, 'w') as f:
        page_url = []

        for expert in expert_page:
            page_index = 1
            while True:
                time.sleep(uniform(0.01, 1.0))
                driver.get(f"{expert.strip()}&sort={_sort_kind}&section=kin&page={page_index}")
                html = driver.page_source
                soup = BeautifulSoup(html, 'html.parser')

                tags = soup.find_all('td', class_="title")
                if not tags:
                    break
                for tag in tags:
                    href = tag.find('a')['href']
                    url = 'https://kin.naver.com' + href
                    page_url.append(url)
                    f.write(url + "\n")
                
                if page_index == max_pages:
                    break
                else:
                    page_index += 1

def main():
    # 파라미터 설정
    geckodriver_path = "C:/VScode/mutimodal/geckodriver"
    useragent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:65.0) Gecko/20100101 Firefox/65.0'
    expert_list_path = "C:/VScode/mutimodal/expert_list.txt"
    output_path = "C:/VScode/mutimodal"
    
    # 드라이버 초기화
    driver = initialize_driver(geckodriver_path, useragent)
    
    # 크롤링 수행
    scrape_expert_responses(driver, expert_list_path, output_path)
    
    # 드라이버 종료
    driver.quit()

if __name__ == "__main__":
    main()
