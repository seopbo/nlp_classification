* 엑소브레인 한국어 BERT 언어모델(KorBERT) 배포 README
* 배포일: 2019-06-03

* 세부 내용
  1. BERT 모델 파라미터
    - 12 layer / 768 hidden / 12 heads
    - 각 폴더 별 bert_config.json 참조
	
  2. BERT 모델 유형
    - 001_bert_morp_pytorch, 002_bert_morp_tensorflow
	  . 형태소분석 결과 기반 BERT 학습 모델
	  . 입력예: ETRI/SL 에서/JKB 한국어/NNP BERT/SL 언어/NNG 모델/NNG 을/JKO 배포/NNG 하/XSV 었/EP 다/EF ./SF
    - 003_bert_eojeol_pytorch, 004_bert_eojeol_tensorflow
	  . 어절 기반 BERT 학습 모델 (형태소분석 미수행)
	  . 입력예: ETRI에서 한국어 BERT 언어 모델을 배포하였다.
	  
  3. tensorflow 및 pytorch 오픈소스
    - tensorflow
	  . https://github.com/google-research/bert
	- pytorch
	  . https://github.com/huggingface/pytorch-pretrained-BERT/
	  . tensorflow로 모델 학습 후, pytorch로 conversion 수행
	  
  4. 형태소분석기 사용
    - OpenAPI 사이트의 형태소분석 API 이용
	  http://aiopen.etri.re.kr/

	  
* 폴더 별 세부 내용
  1. 001_bert_morp_pytorch
    - 형태소분석 결과 기반 BERT 학습 모델 (pytorch 버전)
	- 파일: pytorch 모델 파일 / vocab 파일 / bert_config 파일 / 형태소분석 결과 기반 tokenizer 소스 코드 (pytorch 버전) / MRC 및 문서분류 예제 소스 코드
	
  2. 002_bert_morp_tensorflow
    - 형태소분석 결과 기반 BERT 학습 모델 (tensorflow 버전)
	- 파일: tensorflow 모델 파일 / vocab 파일 / bert_config 파일 / 형태소분석 결과 기반 tokenizer 소스 코드 (tensorflow 버전)

  3. 003_bert_eojeol_pytorch
    - 어절 기반 BERT 학습 모델 (pytorch 버전)
	- 파일: pytorch 모델 파일 / vocab 파일 / bert_config 파일 / 한국어 WordPiece 단위 tokenizer 소스 코드 (pytorch 버전)
	
  4. 004_bert_eojeol_tensorflow
    - 어절 기반 BERT 학습 모델 (tensorflow 버전)
	- 파일: tensorflow 모델 파일 / vocab 파일 / bert_config 파일 / 한국어 WordPiece 단위 tokenizer 소스 코드 (tensorflow 버전)
