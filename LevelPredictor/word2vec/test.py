# 参考 https://github.com/tensorflow/tensorflow/blob/r0.8/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
from six.moves import xrange
from sklearn.manifold import TSNE

from LevelPredictor.word2vec import read_data, build_dataset, plot_with_labels
from LevelPredictor.word2vec.skip_gram_model import generate_batch, generate_graph

if __name__ == '__main__':

    vocabulary_size = 10000

    words = read_data()

    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary_size, words)

    del words  # 释放内存

    batch_size = 128
    embedding_size = 128  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.

    valid_size = 9  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    num_sampled = 64  # Number of negative examples to sample.
    # Input data.
    valid_word = ['主任',
                  '语言表达',
                  '观察',
                  '床头',
                  '护理',
                  '病人',
                  '研究成果',
                  '省部级',
                  '引进']
    valid_examples = [dictionary[li] for li in valid_word]

    batch, labels = generate_batch(data=data, batch_size=batch_size, num_skips=2, skip_window=1)
    for i in range(8):
        print(batch[i], '->', labels[i, 0])
        print(reverse_dictionary[batch[i]], '->', reverse_dictionary[labels[i, 0]])

    graph = generate_graph(batch_size=batch_size,
                           valid_examples=valid_examples,
                           vocabulary_size=vocabulary_size,
                           embedding_size=embedding_size,
                           num_sampled=num_sampled)

    # 开始训练
    num_steps = 30001
    with tf.Session() as session:
        # We must initialize all variables before we use them.
        tf.initialize_all_variables().run()
        print("Initialized")

        average_loss = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips, skip_window)
            feed_dict = {graph['train_inputs']: batch_inputs, graph['train_labels']: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([graph['optimizer'], graph['loss']], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = graph['similarity'].eval()
                for i in xrange(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in xrange(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = graph['normalized_embeddings'].eval()

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels)
