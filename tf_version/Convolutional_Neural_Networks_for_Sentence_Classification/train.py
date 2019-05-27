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


def evaluate(model, dataset, loss_fn, preprocess_fn):
    if tf.keras.backend.learning_phase():
        tf.keras.backend.set_learning_phase(0)

    avg_loss = 0
    for step, mb in tqdm(enumerate(dataset), desc='steps'):
        x_mb, y_mb = preprocess_fn(mb)
        mb_loss = loss_fn(y_mb, model(x_mb))
        avg_loss += mb_loss.numpy()
    else:
        avg_loss /= (step + 1)

    return avg_loss


def main(cfgpath, global_step):
    # parsing config.json
    proj_dir = Path.cwd()
    params = json.load((proj_dir / cfgpath).open())

    # create dataset
    batch_size = params['training'].get('batch_size')
    tr_filepath = params['filepath'].get('tr')
    val_filepath = params['filepath'].get('val')
    tr_ds = create_dataset(tr_filepath, batch_size, True)
    val_ds = create_dataset(val_filepath, batch_size, False)

    # create pre_processor
    vocab = pickle.load((proj_dir / params['filepath'].get('vocab')).open(mode='rb'))
    pre_processor = PreProcessor(vocab=vocab, tokenizer=MeCab().morphs, pad_idx=1)

    # create model

    model = SenCNN(num_classes=2, vocab=vocab)

    # create optimizer & loss_fn
    epochs = params['training'].get('epochs')
    learning_rate = params['training'].get('learning_rate')
    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
    writer = tf.summary.create_file_writer(logdir='./runs/exp')


    # training
    for epoch in tqdm(range(epochs), desc='epochs'):

        tr_loss = 0
        tf.keras.backend.set_learning_phase(1)

        for step, mb in tqdm(enumerate(tr_ds), desc='steps'):
            x_mb, y_mb = pre_processor.convert2idx(mb)

            with tf.GradientTape() as tape:
                mb_loss = loss_fn(y_mb, model(x_mb))
                grads = tape.gradient(target=mb_loss, sources=model.trainable_variables)
            opt.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
            tr_loss += mb_loss.numpy()

            if tf.equal(opt.iterations % global_step, 0):
                with writer.as_default():
                    val_loss = evaluate(model, val_ds, loss_fn, pre_processor.convert2idx)
                    tf.summary.scalar('tr_loss', tr_loss / (step + 1), step=opt.iterations)
                    tf.summary.scalar('val_loss', val_loss, step=opt.iterations)
                    tf.keras.backend.set_learning_phase(1)
        else:
            tr_loss /= (step + 1)

        val_loss = evaluate(model, val_ds, loss_fn, pre_processor.convert2idx)
        tqdm.write('epoch : {}, tr_loss : {:.3f}, val_loss : {:.3f}'.format(epoch + 1, tr_loss, val_loss))

    ckpt_path = proj_dir / params['filepath'].get('ckpt')
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.save(ckpt_path)


if __name__ == '__main__':
    fire.Fire(main)


