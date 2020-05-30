# NLP paper implementation relevant to classification with PyTorch 
The papers were implemented in using korean corpus 

### Single sentence classification (sentiment classification task)
+ Using the [Naver sentiment movie corpus v1.0](https://github.com/e9t/nsmc) (a.k.a. `nsmc`)
+ Configuration
  + `conf/model/{type}.json` (e.g. `type = ["sencnn", "charcnn",...]`)
  + `conf/dataset/nsmc.json`

| Model \ Accuracy | Train (120,000) | Validation (30,000) | Test (50,000) | Date |
| :--------------- | :-------: | :------------: | :------: | :--------------: |
| SenCNN           |  91.78%  |     86.78%     |  85.99%  | 20/03/04 |
| CharCNN          | 85.26% | 81.34% | 81.07% | 20/03/04 |
| ConvRec          | 86.38% | 82.78% | 82.54% | 20/03/23 |
| VDCNN            | 83.85% | 82.23% | 81.61% | 20/03/24 |
| SAN | 90.89% | 86.88% | 86.37% | 20/03/28 |
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
+ Hyper-parameter was arbitrarily selected. (defined by `experiments/base_model/config.json`)

| Model \ Accuracy | Train (6,136) | Validation (682) | Test (758) | Date |
| :--------------- | :-------: | :------------: | :------------: | -------------- |
| Siam     |  93.30%  |     83.57%     |     84.16%     | 19/10/28     |
| SAN | 94.86% | 83.13% | 84.96% | 19/10/28 |
| Stochastic | 88.70% | 81.67% | 81.92% | 19/11/06 |
| ETRIBERT | 95.04% | 93.69% | 93.93% | 19/10/04 |
| SKTBERT | 93.64% | 91.34% | 91.16% | 19/11/10 |


* [x] [A Structured Self-attentive Sentence Embedding](https://github.com/aisolab/nlp_implementation/tree/master/A_Structured_Self-attentive_Sentence_Embedding_ptc) (as SAN)
  + https://arxiv.org/abs/1703.03130
* [x] [Siamese recurrent architectures for learning sentence similarity](https://github.com/aisolab/nlp_implementation/tree/master/Siamese_recurrent_architectures_for_learning_sentence_similarity) (as Siam)
  + https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12195
* [x] [Stochastic Answer Networks for Natural Language Inference](https://github.com/aisolab/nlp_implementation/tree/master/Stochastic_Answer_Networks_for_Natural_Language_Inference) (as Stochastic)
  + https://arxiv.org/abs/1804.07888
* [x] [BERT_pairwise_text_classification](https://github.com/aisolab/nlp_implementation/tree/master/BERT_pairwise_text_classification) (as ETRIBERT, SKTBERT)
  + https://arxiv.org/abs/1810.04805
