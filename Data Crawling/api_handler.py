## api_handler.py
## API를 통해 데이터를 수집하는 함수 모듈

import urllib.request
import urllib.parse
import pandas as pd
from config import CLIENT_ID, CLIENT_SECRET, BASE_URL, DISPLAY_COUNT, MAX_FETCH

## 주어진 query 검색어 리스트를 활용해 데이터를 수집하는 함수
def fetch_data(query):
    kin_list = []
    for q in query:
        start_index = 1
        while start_index <= MAX_FETCH:
            enc_query = urllib.parse.quote(q)
            url = f"{BASE_URL}?query={enc_query}&display={DISPLAY_COUNT}&start={start_index}&sort=sim"
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", CLIENT_ID)
            request.add_header("X-Naver-Client-Secret", CLIENT_SECRET)
            
            with urllib.request.urlopen(request) as response:
                rescode = response.getcode()
                if rescode == 200:
                    response_body = response.read()
                    search_results = eval(response_body.decode('utf-8'))
                    kin_list.extend(search_results['items'])
                else:
                    print(f"Error Code: {rescode}")
            start_index += DISPLAY_COUNT

    ## 수집된 기사 수 출력
    print(f"Collected {len(kin_list)} articles.")
    
    # 데이터프레임으로 변환 & 중복 링크 제거 및 링크 클렌징
    df = pd.DataFrame(kin_list)
    df = df.drop_duplicates(subset='link', keep='first')
    df['link'] = df['link'].apply(lambda x: str(x).replace('\\', ''))
    return df
