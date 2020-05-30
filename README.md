# NLP paper implementation relevant to classification with PyTorch 
The papers were implemented in using korean corpus 

### Single sentence classification (sentiment classification task)
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc) (a.k.a. `nsmc`)
+ Configuration
  + `conf/model/{type}.json` (e.g. `type = ["sencnn", "charcnn",...]`)
  + `conf/dataset/nsmc.json`

| Model \ Accuracy | Train (120,000) | Validation (30,000) | Test (50,000) | Date |
| :--------------- | :-------: | :------------: | :------: | :--------------: |
| SenCNN           |  91.95%  |     86.54%     |  85.84%  | 20/05/30 |
| CharCNN          | 86.29% | 81.69% | 81.38% | 20/05/30 |
| ConvRec          | 86.23% | 82.93% | 82.43% | 20/05/30 |
| VDCNN            | 86.59% | 84.29% | 84.10% | 20/05/30 |
| SAN | 90.71% | 86.70% | 86.37% | 20/05/30 |
| ETRIBERT | 91.12% | 89.24% | 88.98% | 20/05/30 |
| SKTBERT | 92.20% | 89.08% | 88.96% | 20/05/30 |

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
* [x] [BERT_single_sentence_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_single_sentence_classification) (as ETRIBERT, SKTBERT)
  + https://arxiv.org/abs/1810.04805

### Pairwise-text-classification (paraphrase detection task) (under construction)
+ Creating dataset from https://github.com/songys/Question_pair 
+ Configuration
  + `conf/model/{type}.json` (e.g. `type = ["siam", "san",...]`)
  + `conf/dataset/qpair.json`


| Model \ Accuracy | Train (6,136) | Validation (682) | Test (758) | Date |
| :--------------- | :-------: | :------------: | :------------: | -------------- |
| Siam     |  93.00%  |     83.13%     |     83.64%     | 20/05/30     |
| SAN | 89.47% | 82.11% | 81.53% | 20/05/30 |
| Stochastic | 89.26% | 82.69% | 80.07% | 20/05/30 |
| ETRIBERT | 95.07% | 94.42% | 94.06% | 20/05/30 |
| SKTBERT | 95.43% | 92.52% | 93.93% | 20/05/30 |


* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_ptc) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [Siamese recurrent architectures for learning sentence similarity](https://github.com/aisolab/nlp_implementation/tree/master/Siamese_recurrent_architectures_for_learning_sentence_similarity) (as Siam)
  + https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195
* [x] [Stochastic Answer Networks for Natural Language Inference](https://github.com/aisolab/nlp_implementation/tree/master/Stochastic_Answer_Networks_for_Natural_Language_Inference) (as Stochastic)
  + https://arxiv.org/abs/1804.07888
* [x] [BERT_pairwise_text_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_pairwise_text_classification) (as ETRIBERT, SKTBERT)
  + https://arxiv.org/abs/1810.04805
