import pickle
import itertools
import gluonnlp as nlp
from model.utils import Vocab
from sklearn.model_selection import train_test_split

# parsing dataset
with open('./data/train_data', mode='r', encoding='utf-8') as io:
    dataset = []
    data = []

    try:
        while True:
            line = next(io)
            line = line.strip()

            if line:
                data.append(line.split('\t')[1:])
            else:
                dataset.append([list(elm) for elm in zip(*data)])
                data = []
                continue

    except StopIteration:
        print('parsing is done')


label_counter = nlp.data.count_tokens(itertools.chain.from_iterable(map(lambda elm: elm[1], dataset)))
tmp_label_vocab = nlp.Vocab(label_counter, unknown_token=None)
label_vocab = Vocab(tmp_label_vocab.idx_to_token, unknown_token=None)

with open('./data/label_vocab.pkl', mode='wb') as io:
    pickle.dump(label_vocab, io)

tr, val = train_test_split(dataset, test_size=.1, random_state=777)
token_counter = nlp.data.count_tokens(itertools.chain.from_iterable(map(lambda elm: elm[0], tr)))
tmp_token_vocab = nlp.Vocab(token_counter, min_freq=10)
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
tmp_token_vocab.set_embedding(ptr_embedding)
token_vocab = Vocab(tmp_token_vocab.idx_to_token)
token_vocab.embedding = tmp_token_vocab.embedding.idx_to_vec.asnumpy()

with open('./data/token_vocab.pkl', mode='wb') as io:
    pickle.dump(token_vocab, io)
with open('./data/train.pkl', mode='wb') as io:
    pickle.dump(tr, io)
with open('./data/validation.pkl', mode='wb') as io:
    pickle.dump(val, io)
