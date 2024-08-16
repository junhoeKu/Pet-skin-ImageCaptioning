## data_processing.py
## 데이터를 처리하고 저장하는 모듈

import pandas as pd

## 데이터 처리하고 csv 파일로 저장하는 코드
def process_data(df):
    df = df[df.answer.notnull()]
    df.title = df.title.replace('<b>', '', regex = True)
    df.title = df.title.replace(r'<\\/b>', '', regex = True)
    df.question = df.question.apply(lambda x : x.replace('\n', ''))
    df.answer = df.answer.apply(lambda x : x.replace('\n', ''))
    df = df.loc[(df.answer != '답변 없음') & (df.popup == '팝업 없음') & (df.images != '이미지 없음')]

    ## explode 함수를 사용하여 리스트의 각 항목을 별도의 행으로 분리   
    df['answer'] = df['answer'].str.split('|')
    df['images'] = df['images'].str.split('|')
    df = df.explode('answer')
    df = df.explode('images')
    df.to_csv('output.csv', index=False, encoding = 'utf-8')
