# NLP paper implementation with PyTorch
The papers were implemented in using korean corpus 

### Classification
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc)
+ Hyper-parameter was arbitrarily selected. (epoch: 5, mini_batch: 128, except KoBERT (epoch: 2, mini_batch: 32))

| Model \ Accuracy | Train (120,000) | Validation (30,000) | Test (50,000) | Date |
| :--------------- | :-------: | :------------: | :------: | :--------------: |
| SenCNN           |  91.97%  |     86.78%     |  86.00%  | 191004 |
| CharCNN          | 86.54% | 82.25% | 81.89% | 190918 |
| ConvRec          | 88.94% | 83.90% | 83.64% | 190918 |
| VDCNN            | 87.05% | 84.42% | 84.09% | 190918 |
| SAN | 91.17% | 86.54% | 86.00% | 190919 |
| KoBERT | 93.91% | 89.80% | 89.49% | 190920 |

* [x] [Convolutional Neural Networks for Sentence Classification](https://github.com/aisolab/nlp_implementation/tree/master/Convolutional_Neural_Networks_for_Sentence_Classification) (as SenCNN)
  + https://arxiv.org/abs/1408.5882
* [x] [Character-level Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Character-level_Convolutional_Networks_for_Text_Classification) (as CharCNN)
  + https://arxiv.org/abs/1509.01626
* [x] [Efficient Character-level Document Classification by Combining Convolution and Recurrent Layers](https://github.com/aisolab/nlp_implementation/tree/master/Efficient_Character-level_Document_Classification_by_Combining_Convolution_and_Recurrent_Layers) (as ConvRec)
  + https://arxiv.org/abs/1602.00367
* [x] [Very Deep Convolutional Networks for Text Classification](https://github.com/aisolab/nlp_implementation/tree/master/Very_Deep_Convolutional_Networks_for_Text_Classification) (as VDCNN)
  + https://arxiv.org/abs/1606.01781
* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_cls) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [BERT_single_sentence_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_single_sentence_classification) (as KoBERT)
  + https://arxiv.org/abs/1810.04805

### Paraphrase detection
+ Creating dataset from https://github.com/songys/Question_pair 
+ Hyper-parameter was arbitrarily selected. (epoch: 5, mini_batch: 64)

| Model \ Accuracy | Train (6,060) | Validation (1,516) | Date |
| :--------------- | :-------: | :------------: | -------------- |
| SAN           |  91.93%  |     81.46%     |          |
| Siam | 92.98% | 84.03% | 191004 |
| KoBERT | 90.80% | 93.33% |  |


* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_ptc) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [Siamese recurrent architectures for learning sentence similarity](https://github.com/aisolab/nlp_implementation/tree/master/Siamese_recurrent_architectures_for_learning_sentence_similarity) (as Siam)
  + https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195
* [ ] Stochastic Answer Networks for Natural Language Inference
  + https://arxiv.org/abs/1804.07888
* [x] [BERT_pairwise_text_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_pairwise_text_classification) (as KoBERT)
  + https://arxiv.org/abs/1810.04805

### Language model
* [ ] Character-Aware Neural Language Models
  + https://arxiv.org/abs/1508.06615


### Named entity recognition

+ Using the [Naver nlp-challange corpus for NER](https://github.com/naver/nlp-challenge/tree/master/missions/ner)
+ Hyper-parameter was arbitrarily selected.
* [x] [Bidirectional LSTM-CRF Models for Sequence Tagging](https://github.com/aisolab/nlp_implementation/tree/master/Bidirectional_LSTM-CRF_Models_for_Sequence_Tagging)
	+ https://arxiv.org/abs/1508.01991
* [ ] End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
	+ https://arxiv.org/abs/1603.01354
* [ ] Neural Architectures for Named Entity Recognition
	+ https://arxiv.org/abs/1603.01360
* [ ] BERT_single_sentence_tagging
	+ https://arxiv.org/abs/1810.04805


### Neural machine translation

| Model \ Perplexity | Train () | Validation  () | Test () | Date |
| ------------------ | -------- | -------------- | ------- | ---- |
| LuongAttn          |          |                |         |      |
| Transformer        |          |                |         |      |

* [x] Effective Approaches to Attention-based Neural Machine Translation (as LuongAttn)
	+ https://arxiv.org/abs/1608.07905
* [ ] Attention Is All You Need (as Transformer)
	+ https://arxiv.org/abs/1706.03762


### Machine reading comprension
* [ ] Machine Comprehension Using Match-LSTM and Answer Pointer
	+ https://arxiv.org/abs/1611.01603
* [ ] Bi-directional attention flow for machine comprehension
	+ https://arxiv.org/abs/1611.01603
* [ ] BERT_question_answering
	+ https://arxiv.org/abs/1810.04805