## main.py
## 전체 프로세스를 실행하는 메인 파일

from api_handler import fetch_data
from selenium_handler import crawl_webpage
from data_processing import process_data

def main():
    queries = ['고양이 피부', '고양이 피부병', '고양이 두드러기', '고양이 여드름', '고양이 비듬', '고양이 각질', '고양이 농포', '고양이 진드기', '고양이 결절', '고양이 종괴']
    kin_df = fetch_data(queries)
    kin_crawled_data = crawl_webpage(kin_df['link'].tolist())
    
    # API에서 가져온 데이터프레임과 셀레니움에서 크롤링한 데이터프레임을 병합
    final_df = kin_df.merge(kin_crawled_data, on='link', how='inner')
    
    # 최종 데이터 처리
    process_data(final_df)

if __name__ == "__main__":
    main()
