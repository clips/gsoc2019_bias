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

    def _replace_at_pos(self, sentence, pos, word):
        new_sentence = sentence.copy()
        new_sentence[pos] = word
        return new_sentence

    def _select_replacement(self, sent_current, sent_original, position, replacements):
        if len(replacements) == 0:
            return sent_current

        new_sentences = [self._replace_at_pos(sent_current, position, word) for word in replacements]
        new_predictions = [self.model.predict(sentence) for sentence in new_sentences]

    def _perturb(self, sent_current, sent_original, position, target=None):
        sent_len = np.sum(np.sign(sent_current))
        assert position < sent_len

        word_original = sent_current[position]
        replacements = self.replacement_matrix.get_replacements(word_original)
        return self._select_replacement(sent_current, sent_original, position, replacements)

    def attack(self, sentence, target_label=None):
        self.target = None
        self.original_prediction = np.argmax(self.model.predict(sentence))

        sent_len = np.sum(np.sign(sentence.copy()))

        mod_sentence = sentence.copy()
        for i in range(sent_len):
            word = mod_sentence[i]
            word_new = self.perturb(mod_sentence, i, sentence, target_label)