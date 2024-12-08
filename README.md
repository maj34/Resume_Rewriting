# 취업 준비생을 위한 자기소개서 평가 및 첨삭 프레임워크
GPT-4o를 활용한 자기소개서 평가와 EEVE-Korean-10.8B-v1.0 기반 분야별 모범 자기소개서 Fine-tuning을 통한 맞춤형 첨삭 프레임워크

</br>

## 0. Prerequisites
- **requirements.txt**: 프로젝트 실행 전에 설치해야 하는 라이브러리가 명시된 파일
  ```bash
  pip install -r requirements.txt
  ```
- **eda.ipynb**: 데이터셋(자기소개서) 분포를 학과별로 분석하는 주피터 노트북 파일
- **crawling.py**: 링커리어에서 합격 자기소개서를 자동으로 크롤링하는 코드
- **clean_resume_filtering.ipynb**: 크롤링된 자기소개서 중 직무 적합성 점수가 4점 이상인 모범 자기소개서만 필터링하는 코드 (data 폴더 내 filtered_data_{분야}.csv 제공)

</br>

## 1. GPT-4o based Resume Evaluation
- **evaluation.py**: 사용자가 입력한 자기소개서에 대하여 질문 관련도(Relevance), 의도 명확성(Clarity), 직무 적합성(Compatibility), 정보 구체성(Concreteness) 4가지 평가 기준을 기반으로 1~5점 리커트 척도로 평가
- 실행 전 OpenAI API key를 코드 내 변수에 직접 입력하거나 환경변수로 설정
- 각 Category 별로 평가할 수 있도록 Argument에 ['공학', '자연', '인문', '사회', '기타']를 각각 입력

  ```bash
  sh src/evaluation.sh
  ```

</br>

## 2. Fine-Tuning Exemplary Resume
- **fine_tuning_with_qlora.py**: EEVE-Korean-10.8B-v1.0 모델을 QLoRA 기법을 사용해 Fine-tuning하는 코드

- **fine_tuning_{분야}.sh**:
공학(engineering), 인문(humanities), 자연(natural), 사회(social), 일반(total) 총 5가지 분야별로 모델을 학습하기 위한 쉘 스크립트
실행 시 checkpoint/EEVE-Korean-10.8B-v1.0/{분야} 폴더 하위에 추론용 체크포인트 생성
  ```bash
  sh src/fine_tuning_engineering.sh
  ```

</br>

## 3. Resume Editing (Inference)
- **inference.py**:
입력한 자기소개서를 바탕으로 첨삭된 자기소개서를 생성하는 코드
사전에 Fine-tuning 된 체크포인트 활용

- **inference.sh**:
공학(engineering), 인문(humanities), 자연(natural), 사회(social), 일반(total) 각 분야별로 첨삭 추론을 실행하는 쉘 스크립트
  ```bash
  sh src/inference.sh engineering "자기소개서 내용..."
  ```

