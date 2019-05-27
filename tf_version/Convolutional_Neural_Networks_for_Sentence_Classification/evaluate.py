import json
import tensorflow as tf
import fire
import pickle
from model.net import SenCNN
from model.utils import PreProcessor
from pathlib import Path
from mecab import MeCab
from tqdm import tqdm


def create_dataset(filepath, batch_size, shuffle=True, drop_remainder=True):
    ds = tf.data.TextLineDataset(filepath)
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)

    ds = ds.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return ds


def get_accuracy(model, dataset, preprocess_fn):
    if tf.keras.backend.learning_phase():
        tf.keras.backend.set_learning_phase(0)

    correct_cnt = 0
    total_cnt = 0

    for step, mb in tqdm(enumerate(dataset), desc='steps'):
        x_mb, y_mb = preprocess_fn(mb)
        score_mb = model(x_mb)
        y_hat_mb = tf.argmax(score_mb, axis=1)
        correct_cnt += sum(y_mb.numpy() == y_hat_mb.numpy())
        total_cnt += y_mb.get_shape()[0]
    else:
        acc = correct_cnt / total_cnt
    return acc


def main(cfgpath):
    # parsing config.json
    proj_dir = Path.cwd()
    params = json.load((proj_dir / cfgpath).open())

    # create dataset
    batch_size = params['training'].get('batch_size')
    tr_filepath = params['filepath'].get('tr')
    val_filepath = params['filepath'].get('val')
    tst_filepath = params['filepath'].get('tst')

    tr_ds = create_dataset(tr_filepath, batch_size, False, False)
    val_ds = create_dataset(val_filepath, batch_size, False, False)
    tst_ds = create_dataset(tst_filepath, batch_size, False, False)

    # create pre_processor
    vocab = pickle.load((proj_dir / params['filepath'].get('vocab')).open(mode='rb'))
    pre_processor = PreProcessor(vocab=vocab, tokenizer=MeCab().morphs, pad_idx=1)

    # create model
    model = SenCNN(num_classes=2, vocab=vocab)
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(save_path=tf.train.latest_checkpoint(proj_dir / 'checkpoint'))

    # evluation
    tr_acc = get_accuracy(model, tr_ds, pre_processor.convert2idx)
    val_acc = get_accuracy(model, val_ds, pre_processor.convert2idx)
    tst_acc = get_accuracy(model, tst_ds, pre_processor.convert2idx)

    print('tr_acc: {:.2%}, val_acc : {:.2%}, tst_acc: {:.2%}'.format(tr_acc, val_acc, tst_acc))


if __name__ == '__main__':
    fire.Fire(main)