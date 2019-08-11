from typing import List

import numpy as np

class GenderSwitchAttackGenetic():
    def __init__(self, model, samples, replacement_matrix,
                 population_size = 20, iterations = 100, use_language_model = False,
                 language_model_threshold = 0.1, language_model_contex = 1):

        self.model = model
        self.samples = samples
        self.replacement_matrix = replacement_matrix

        self.population_size = population_size
        self.iterations = iterations
        self.use_language_model = use_language_model
        self.language_model_threshold = 0.1
        self.language_model_context = 1

    def _replace_at_pos(self, sentence, pos, word):
        new_sentence = sentence.copy()
        new_sentence[pos] = word
        return new_sentence

class GenderSwitchAttackBaseline():
    def __init__(self, model, samples, replacement_matrix,
                 use_language_model=False, language_model_threshold=0.1, language_model_contex=1):
        self.model = model
        self.samples = samples
        self.replacement_matrix = replacement_matrix

        self.use_language_model = use_language_model
        self.language_model_threshold = 0.1
        self.language_model_context = 1

        self.target = None
        self.original_prediction = None
        self.current_prediction = None

    def attack_all(self):
        for sample in self.samples:
            self.attack(sample)

    #Routine that executes a baselike perturbation on the $sentence.
    #If a target label is provided then the goal is to maximize that label's probability prediction,
    #otherwise, the algorithm attempts to minimize the probability prediction of the original label
    def attack(self, sentence : str, target_label : int = None):
        #Save the original and current prediction probability arrays
        print("Attacking sentence " + sentence)
        self._set_orig_prediction(self.model.predict_one(sentence, plain = True))
        print("Original prediction: {}".format(self.original_prediction))
        self._set_curr_prediction(self.original_prediction)

        #If target is provided, use it.
        if target_label is not None:
            print("Maximize label: {}".format(target_label))
            self._set_target(target_label)
        #Otherwise minimize original label.
        else:
            print("Minimize label: {}".format(np.argmax(self.original_prediction)))
            self._set_target(-(np.argmax(self.original_prediction) + 1))

        #Split the input to individual words
        current_sentence : List(str) = sentence.split(' ')

        #Save the modifications to a list
        modifications = list()
        for index in range(len(current_sentence)):
            orig_word = current_sentence[index]
            print("Modifying word: " + orig_word)

            new_word, new_prediction = self._perturb(current_sentence, index)
            if new_word is not None:
                print("New word: " + new_word)
                print(new_prediction)
                #Replace the word with the new word
                current_sentence = self._replace_at_pos(current_sentence, index, new_word)

                self._set_curr_prediction(new_prediction)
                modifications.append((orig_word, new_word))

    #With an input list of sentences, determine the one that best approaches the currently stored target
    def _select_best(self, sentence, position, words):
        best = -1
        print(sentence)
        print(position)
        print(words)
        testers = [self._replace_at_pos(sentence, position, word) for word in words]
        print(testers)
        predictions = [self.model.predict_one(' '.join(tester), plain = True) for tester in testers]
        print(predictions)

        for index in range(len(words)):
            if self.target < 0:
                if(predictions[index][-(self.target + 1)] < self.current_prediction[-(self.target + 1)]):
                    best = index
            else:
                if(predictions[index][self.target] > self.current_prediction[self.target]):
                    best = index

        if best != -1:
            return words[best], predictions[best]
        else:
            return None, None

    #With an input sentence and position, generate replacements and return the best new word and the new prediction
    def _perturb(self, sent_current, position):
        replacements = self.replacement_matrix.get_replacements(word=sent_current[position].lower())
        return self._select_best(sent_current, position, replacements)

    #Returns an altered sentence with the word at the $pos index changed with $word
    def _replace_at_pos(self, sentence, pos, word):
        new_sentence = sentence.copy()
        new_sentence[pos] = word
        return new_sentence

    def _set_target(self, target):
        self.target = target

    def _set_orig_prediction(self, prediction):
        self.original_prediction = prediction

    def _set_curr_prediction(self, prediction):
        self.current_prediction = prediction