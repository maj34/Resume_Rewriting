# 취업 준비생을 위한 자기소개서 평가 및 첨삭 프레임워크
이 저장소는 GPT-4o 기반 자기소개서 평가와 EEVE-Korean-10.8B-v1.0 모델을 사용하여 분야 별 모범 자기소개서 Fine-tuning을 통해 자기소개서를 첨삭하는 프레임워크 예시 코드입니다.

</br>

## 0. 사전 준비
- **requirements.txt**: 프로젝트 실행 전에 설치해야 하는 라이브러리가 명시된 파일입니다.  
  ```bash
  pip install -r requirements.txt
  ```

- **eda.ipynb**: 데이터셋(자기소개서) 분포를 학과별로 분석하는 주피터 노트북 파일입니다. 한글 폰트로 시각화를 원한다면 font 폴더 내 NanumSquare 폰트를 활용하세요.
- **crawling.py**: 링커리어에서 합격 자기소개서를 자동으로 수집(크롤링)하는 코드입니다. 프로젝트 실행 시 이미 정제된 CSV가 제공되므로 반드시 실행할 필요는 없습니다.
- **clean_resume_filtering.ipynb**: 크롤링된 자기소개서 중 직무적합성 점수가 4점 이상인 모범 자기소개서만 필터링하는 코드입니다. data 폴더 내 filtered_data_{분야}.csv를 이미 제공하므로 별도 실행 없이도 사용 가능합니다.

</br>

## 1. 자기소개서 평가 (GPT-4o 기반)
- **evaluation.py**:
사용자가 입력한 자기소개서에 대하여 질문 관련도(Relevance), 의도 명확성(Clarity), 직무 적합성(Compatibility), 정보 구체성(Concreteness) 4가지 Aspects를 기반으로 1~5점 리커트 척도로 평가합니다.
- 실행 전 OpenAI API key를 코드 내 변수에 직접 입력하거나 환경변수로 설정해야 합니다.
- 각 Category 별로 평가할 수 있도록 Argument에 ['공학', '자연', '인문', '사회', '기타']를 각각 입력하면 됩니다.

  ```bash
  python src/evaluation.py --category '공학'
  ```

</br>

## 2. 모범 자기소개서 학습 (Fine-tuning)
- **fine_tuning_with_qlora.py**: EEVE-Korean-10.8B-v1.0 모델을 QLoRA 기법을 사용해 Fine-tuning하는 코드입니다.

- **fine_tuning_{분야}.sh**:
공학(engineering), 인문(humanities), 자연(natural), 사회(social), 일반(total) 총 5가지 분야별로 모델을 학습하기 위한 스크립트입니다.
실행 시 checkpoint/EEVE-Korean-10.8B-v1.0/{분야} 폴더 하위에 추론용 체크포인트가 생성됩니다.
  ```bash
  sh src/fine_tuning_engineering.sh
  ```

</br>

## 3. 자기소개서 첨삭 (Inference)
- **inference.py**:
입력한 자기소개서를 바탕으로 첨삭된 자기소개서를 생성하는 코드입니다.
사전에 Fine-tuning 된 체크포인트를 활용합니다.

- **inference.sh**:
공학(engineering), 인문(humanities), 자연(natural), 사회(social), 일반(total) 각 분야별로 첨삭 추론을 실행하는 쉘 스크립트입니다.
  ```bash
  sh src/inference.sh engineering "자기소개서 내용..."
  ```




