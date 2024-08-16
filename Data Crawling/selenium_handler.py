## selenium_handler.py
## 셀레니움을 이용한 웹 크롤링 기능 구현

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.alert import Alert
from selenium.common.exceptions import NoAlertPresentException
import time
import random
import pandas as pd

def crawl_webpage(links):
    driver = webdriver.Chrome()
    driver.implicitly_wait(3)
    kin_data = []

    for index, link in enumerate(links):
        row_data = {'link': link, 'popup': '팝업 없음', 'question': None, 'images': None, 'answer': None}
        try:
            driver.get(link)
            time.sleep(random.uniform(0.5, 1))
            
            ## 팝업 창 처리
            try:
                Alert(driver).accept()
                row_data['popup'] = '팝업 있음'
                continue
            except NoAlertPresentException:
                pass

            ## 질문 수집
            questions = driver.find_elements(By.CLASS_NAME, 'questionDetail')
            row_data['question'] = questions[0].text if questions else '사라진 페이지'
            
            ## 이미지 URL 수집
            images = driver.find_elements(By.TAG_NAME, 'img')
            image_urls = [img.get_attribute('src') for img in images if 'w750' in img.get_attribute('src')]
            row_data['images'] = ' | '.join(image_urls) if image_urls else '이미지 없음'
            
            ## 답변 수집
            answers = driver.find_elements(By.CLASS_NAME, 'se-component-content')
            filtered_answers = [answer.text for answer in answers if '.com' not in answer.text]
            row_data['answer'] = ' | '.join(filtered_answers) if filtered_answers else '답변 없음'

            kin_data.append(row_data)
            
            ## 진행 상황 출력
            progress = (index + 1) / len(links) * 100
            print(f"Progress: {progress:.2f}% Complete")
        except Exception as e:
            print(f"Error processing link {link}: {str(e)}")

    driver.quit()
    return pd.DataFrame(kin_data)
