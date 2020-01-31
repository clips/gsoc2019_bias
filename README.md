# Gender Bias in Language Processing 
### Google Summer of Code 2019

By: Panagiotis Lantavos, Artistotle University of Thessaloniki

Mentor: Madhumita Sushil

Organisation: Computational Linguistics & Psycholinguistics Research Centre, University of Antwerp (CLiPS)

## Abstract

The goal of this project is the implementation of an algorithm/framework which is able to establish and provide metrics on the presence of gender bias or sexism in natural language processing models. The core concept is the alteration of gendered words in the input text set based on the relation of those words in the embedding space.

For example, the word man may be altered to woman, the word father may be altered to mother and the word priest may be altered to priestess.

During the course of the project, an algorithm was implemented which applies these alterations in order to produce adversarial examples for the given model. The vulnerability of the model to this adversarial attack is used to determine the presence of gender bias. The implementation and results of the implemented algorithm are described below, along with further work to be done.

## Implementation

This project involves three core components: The use of word embeddings for semantic word transformation, the use of language modeling for small-context syntax checking and the use of a genetic algorithm for the application of these alterations. 

To function, the project does not require any access to the model parameters, but needs to be able to access the model's predict function and that function needs to provide probability distributions, rather than just the final label.

##### Embeddings and Alterations

For the embedding implementation, the project includes loaders and functions that can load GLoVE formatted embedding files such as those found in https://nlp.stanford.edu/projects/glove/. For the purposes of generating the alterations, the 840B token/300  dimension embeddings were used. 

The alterations are stored in a dictionary where for each word, n words that comply with the required function in the embeddings space are stored. 

Here the required function is the addition or subtraction of other word vectors in order to achieve different semantic meaning. For the case of gender, man>woman and woman>man are used.

The function that is applied in the embedding space is provided as a parameter and can be any addition or subtraction of other word vectors. For the presence of gender bias the two alterations (+woman-man) and (+man-woman) are used. For finding the closest words after the function is applied cosine similarity is used.

##### Language Modeling

A goal of this project is that the adversarial examples generated are human-readable and make sense syntactically. To ensure this, language modeling was used to filter out alterations that don't make sense in the context.

For this purpose a pretrained model on the LM-1B dataset was used. The model is the one found in the tensorflow research models directory found at https://github.com/tensorflow/models/tree/master/research/lm_1b. With the current implementation, the language modeling component filters out word alterations that score below a certain threshold in comparison to the original word. The threshold used for tests is 25% since it was observed that even words that fit within a certain context might hold quite different probability scores in the language model.

Currently, the language model only uses the *n* closest prefix words from the target, although an improvement would be to use suffix as well. 

##### Genetic Algorithm

For the selection of optimal alterations to attack the target model, a genetic algorithm is used. The algorithm creates a starting population by selecting a random word in the input text to be changed and then attempting to alter that word. This proceeds until a word has been successfully altered or the individual is discarded. 

After that, the genetic algorithm proceeds with normal crossover and mutation, with crossover being the mixup of two of the altered sentences at a random point between the first and last word and mutation being the alteration of a random word on a sentence of the population.

The genetic algorithm terminates after a set number of iterations or when it successfully alters the final label that the model predicts.

##### Dataset

Two datasets where used for this project, both regarding abusive language on Twitter, those are the Hate and Abusive Speech on Twitter (Founta et al), which consists of 100.000 tweets annotated through crowdsourcing as abusive, spam, hateful or normal. This dataset was the one used the most and thanks go to the authors for providing the data directly. 

As a secondary dataset, the Automated Hate Speech Detection dataset (Davidson et al), which consists of 25.000 tweets, with similar annotations was used.


##### Further Goals

* Implement a better cutoff algorithm to determine replacement words. Examining different approaches other than plain cosine similarity might be beneficial.
* Test the algorithm on additional models, especially complex models that create decision functions with sharp edges, as research shows that those models are the ones most subject to adversarial perturbations.
* Optimize algorithm performance, especially in relation to language modeling and docker runtimes.
* Use a message queuing system for communication between different modules, instead of dumping all the code in a single container. This will help with both containerization as well as the performance issue above. 
* Provide instructions for use of the algorithm by 3rd parties and facilitate integration.
* Include suffix into language model checking for better syntactical coherence of the generated sentences.

## Results

Sadly, although the results of the algorithm are certainly interesting they don’t quite meet the goals of the project. The algorithm produces alterations that are highly volatile, in the sense that although several methods are employed to restrict the changes and enforce that the results make syntactical and logical sense, often they end up not doing so. Additionally, the dataset used is sourced from Twitter, which has its own issues with syntax and grammar, often confusing the language model component. 

Thus, no significant metrics were extracted in the course of the project. Nevertheless there were some results which provide insight into the workings of embeddings and models and are promising for the use of this or a similar algorithm for semantic-based adversarial example generation. 

For example, besides trivial alterations of semantically gendered words such as those described above (man-woman, father-mother, etc), the model often replaced words with synonyms that are gender-charged, such as replacing “beautiful” with “gorgeous” or “fabulous” or “nasty” with “slutty” or “blue” with “pink” in the case of -man+woman transformation. 

These are words that the embeddings indicate are gender related and indeed that was the idea behind using the word embeddings to produce these alterations.

It’s also notable that the algorithm found substantially more label-altering replacements for male to female alterations than the opposite and this fact may indicate the presence of bias in its own right.

Some examples of the alterations produced by the algorithm are shown below: 
```
Current sentence: Just wondered #fishermentokings a documentary on the beautiful photography of Olive Edis. Brilliant and inspiring to watch.
Current prediction: [0.02736139 0.0341036  0.81014938 0.12838562]
Orig word: beautiful
New word: gorgeous
Current sentence: Just wondered #fishermentokings a documentary on the gorgeous photography of Olive Edis. Brilliant and inspiring to watch.
Current prediction: [0.0424355  0.03935942 0.75177873 0.16642635]
Orig word: Brilliant
New word: fabulous
Current sentence: Just wondered #fishermentokings a documentary on the gorgeous photography of Olive Edis. fabulous and inspiring to watch.
Current prediction: [0.04182737 0.0363861  0.69969317 0.22209336]
```
```
Attacking sentence then you ask for a contact picture so you can see them nd they butt ugly. like got damn do I got the ugly girl magnet
Original prediction: [0.71457868 0.05613658 0.18483814 0.0444466 ]
Minimize label: 0
Orig word: picture
New word: pictures
Current sentence: then you ask for a contact pictures so you can see them nd they butt ugly. like got damn do I got the ugly girl magnet
Current prediction: [0.67031302 0.06324703 0.21911846 0.04732149]
Orig word: ugly
New word: unattractive
Current sentence: then you ask for a contact pictures so you can see them nd they butt unattractive. like got damn do I got the ugly girl magnet
Current prediction: [0.52787926 0.05803266 0.35984719 0.05424088]
Orig word: damn
New word: darn
Current sentence: then you ask for a contact pictures so you can see them nd they butt unattractive. like got darn do I got the ugly girl magnet
Current prediction: [0.25884776 0.05492756 0.63143423 0.05479045]
Orig word: ugly
New word: unattractive
Current sentence: then you ask for a contact pictures so you can see them nd they butt unattractive. like got darn do I got the unattractive girl magnet
Current prediction: [0.1595698  0.04629717 0.72820007 0.06593296]
Orig word: girl
New word: teen
Current sentence: then you ask for a contact pictures so you can see them nd they butt unattractive. like got darn do I got the unattractive teen magnet
Current prediction: [0.1207891  0.04348537 0.75958554 0.07614   ]
Orig word: magnet
New word: pin
Current sentence: then you ask for a contact pictures so you can see them nd they butt unattractive. like got darn do I got the unattractive teen pin
Current prediction: [0.09621861 0.04254847 0.75989746 0.10133545]
```

## References

Alzantot et al, EMNLP 2018, Generating Natural Language Adversarial Examples, https://arxiv.org/pdf/1804.07998v2.pdf

Founta at al, Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior, https://arxiv.org/pdf/1802.00393.pdf

Davidson et al, ICWSM 2017, Automated Hate Speech Detection and the Problem of Offensive Language, https://aaai.org/ocs/index.php/ICWSM/ICWSM17/paper/view/15665/14843
