import tensorflow as tf
import numpy as np

from src.utils.language_model import language_model_utils, language_model_loader

class LM(object):
    def __init__(self):
        self.PBTXT_PATH = 'resources/goog_lm/graph-2016-09-10.pbtxt'
        self.CKPT_PATH = 'resources/goog_lm/ckpt-*'
        self.VOCAB_PATH = 'resources/goog_lm/vocab-2016-09-10.txt'

        self.BATCH_SIZE = 1
        self.NUM_TIMESTEPS = 1
        self.MAX_WORD_LEN = 50

        self.vocab = language_model_loader.CharsVocabulary(self.VOCAB_PATH, self.MAX_WORD_LEN)
        print('LM vocab loading done')
        with tf.device("/gpu:1"):
            self.graph = tf.Graph()
            self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.t = language_model_utils.LoadModel(self.sess, self.graph, self.PBTXT_PATH, self.CKPT_PATH)

    def get_words_probs(self, prefix_words, list_words, suffix=None):
        targets = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.int32)
        weights = np.ones([self.BATCH_SIZE, self.NUM_TIMESTEPS], np.float32)

        if prefix_words.find('<S>') != 0:
            prefix_words = '<S> ' + prefix_words
        prefix = [self.vocab.word_to_id(w) for w in prefix_words.split()]
        prefix_char_ids = [self.vocab.word_to_char_ids(w) for w in prefix_words.split()]

        char_ids_inputs = np.zeros([self.BATCH_SIZE, self.NUM_TIMESTEPS, self.vocab.max_word_length], np.int32)

        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        inputs = [[samples[-1]]]
        char_ids_inputs[0, 0, :] = char_ids_samples[-1]
        softmax = self.sess.run(self.t['softmax_out'],
                                feed_dict={
                                    self.t['char_inputs_in']: char_ids_inputs,
                                    self.t['inputs_in']: inputs,
                                    self.t['targets_in']: targets,
                                    self.t['target_weights_in']: weights
                                })

        words_ids = [self.vocab.word_to_id(w) for w in list_words]
        word_probs = [softmax[0][w_id] for w_id in words_ids]
        word_probs = np.array(word_probs)

        if suffix is None:
            suffix_probs = np.ones(word_probs.shape)
        else:
            suffix_id = self.vocab.word_to_id(suffix)
            suffix_probs = []
            for idx, w_id in enumerate(words_ids):
                inputs = [[w_id]]
                w_char_ids = self.vocab.word_to_char_ids(list_words[idx])
                char_ids_inputs[0, 0, :] = w_char_ids
                softmax = self.sess.run(self.t['softmax_out'],
                                        feed_dict={
                                            self.t['char_inputs_in']: char_ids_inputs,
                                            self.t['inputs_in']: inputs,
                                            self.t['targets_in']: targets,
                                            self.t['target_weights_in']: weights
                                        })
                suffix_probs.append(softmax[0][suffix_id])
            suffix_probs = np.array(suffix_probs)
        return suffix_probs * word_probs


if __name__ == '__main__':
    my_lm = LM()
    list_words = 'play will playing played afternoon'.split()
    prefix = 'i'
    suffix = 'yesterday'
    probs = (my_lm.get_words_probs(prefix, list_words, suffix))
    for i in range(len(list_words)):
        print(list_words[i], ' - ', probs[i])

    list_words = 'nice great game weather motorcycle'.split()
    prefix = 'this morning is'
    probs = (my_lm.get_words_probs(prefix, list_words))
    for i in range(len(list_words)):
        print(list_words[i], ' - ', probs[i])


    list_words = 'son daughter car weather him afternoon'.split()
    prefix = 'their'
    suffix = 'is going to go to college'
    probs = (my_lm.get_words_probs(prefix, list_words, suffix))
    for i in range(len(list_words)):
        print(list_words[i], ' - ', probs[i])