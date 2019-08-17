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

    def attack_all(self, metric = True):
        sentences = []
        perturbed = []
        orig_labels = []
        pert_labels = []
        num_changes = []
        label_changed = []
        for sample in self.samples:
            sentences.append(sample)
            sent, modifications = self.attack(sample)
            perturbed.append(sent)
            if np.argmax(self.original_prediction) != np.argmax(self.current_prediction):
                print("Successful perturbation")
            orig_labels.append(np.argmax(self.original_prediction))
            print(orig_labels[-1])
            pert_labels.append(np.argmax(self.original_prediction))
            print(pert_labels[-1])
            label_changed.append(1 if orig_labels[-1] != pert_labels[-1] else 0)
            print(label_changed[-1])

            num_changes.append(len(modifications))

        if metric:
            percent = sum(label_changed)/len(self.samples)
            print("Successful modifications = {}, {}".format(sum(label_changed), percent))

            avg_modifications = sum(num_changes)/len(self.samples)
            mod_weight = 0
            for i in range(len(self.samples)):
                print("Modifications {}/{}={}, success = {}".format(avg_modifications, num_changes[i], avg_modifications/num_changes[i], label_changed[i]))

                if(label_changed[i] == 1):
                    mod_weight += avg_modifications[i]/num_changes[i]
                else:
                    mod_weight -= avg_modifications[i]/num_changes[i]
            print(mod_weight)

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

            new_word, new_prediction = self._perturb(current_sentence, index)
            if new_word is not None:
                print("Orig word: " + orig_word)
                print("New word: " + new_word)
                #Replace the word with the new word
                current_sentence = self._replace_at_pos(current_sentence, index, new_word)
                print("Current sentence: {}".format(' '.join(current_sentence)))
                print("Current prediction: {}".format(new_prediction))

                self._set_curr_prediction(new_prediction)
                modifications.append((orig_word, new_word))

        return current_sentence, modifications

    #With an input list of sentences, determine the one that best approaches the currently stored target
    def _select_best(self, sentence, position, words):
        best = -1
        testers = [self._replace_at_pos(sentence, position, word) for word in words]
        predictions = [self.model.predict_one(' '.join(tester), plain = True) for tester in testers]

        #The difference must be at least 5% for a change to be made
        for index in range(len(words)):
            if self.target < 0:
                if(predictions[index][-(self.target + 1)] < self.current_prediction[-(self.target + 1)]*0.95):
                    best = index
            else:
                if(predictions[index][self.target] > self.current_prediction[self.target]*1.05):
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