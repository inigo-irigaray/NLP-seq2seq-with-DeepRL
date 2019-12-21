# Leveraging the power of Deep Reinforcement Learning training NLP algorithms


**<div style=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue systems and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.**</div>

-------
**1. Preliminaries.** Introduces a conceptual background on the NLP literature and state-of-the-art algorithms for conversational modelling, machine translation and other key challenges in the field; as well as the BLEU metric against which the model will be evaluated.

**2. seq2seq with Cross-Entropy & REINFORCE - the algorithms.** Details the specifics of the algorithms used for this particular experiment and the core structure of the approximation models employed.

**3. Training.** Analyzes the progress, duration and statistics of the two different training methods until halting.

**4. Results & Discussion.** Tests the chatbot agent generated from the model in the free open Telegram environment.

**5. Future work.** Explores potential avenues of interest for future experiments.


---------
## 1. Preliminaries

<div style=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The recent advancements on deep neural network architectures, sustained on improved computational capabilities and larger, more comprehensive datasets, has propelled an enormous amount of success in the field of Machine Learning. This, coupled with better systems for vectorial representation of language structures in the form of embeddings, has put Natural Language Processing at the forefront of research and progress. The following subsections serve as an overview of major methods for different NLP tasks and the works that led to this implementation of seq2seq based on Ranzato et al.</div>

#### Embeddings

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Embeddings are distributional vectors representing different levels of linguistic structures (characters and words). They capture meaning by encoding reference attributes to each structure based on the context in which it apperas, i.e. the other words and characters that tend to appear next to the target structure [1-Ch6].

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Word2vec** are some of the most popular word embedding algorithms. The **skip-gram** model predicts context words based on the target word. It trains a logistic regression classifier that computes the conditional probability between pairs of words with the dot product between their embeddings. Opposite to skip-gram, the continuous bag-of-words (**CBOW**) predicts a target word from the context words. Based on fully-connected NNs with one hidden-layer, these methods allow for efficient representations of dense vectors that capture semantic and syntactic information. However, they are weak on sentiment, polisemy, phrase meaning, conceptual meaning and out-of-vocabulary (OOV) words, which is problematic for some tasks [2, 1-Ch6].

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contextualized word embeddings are another type of embeddings that directly address some of these issues. **ELMo** creates a different word embedding for each context in which a word appears, thus capturing polisemic meaning. It consists of a bidirectional language model of a forward Long Short-Term Memory (LSTM) network to model the joint probability of a series of input tokens and predict the next token, a backward LSTM network that predicts the previous token and the cross-entropy loss between the two predictions. **OpenAI-GPT-2, BERT**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Alternatively, approaching the embedding problem at the character-level has allowed to researchers to tackle some issues aforementioned, like OOVs, and tasks like named-equity recognition (NER), adding meaning to phrases by representing words simply as a combination of characters. Additionally, they prove more effective with some morphologically-rich languages like Spanish, and languages where text is composed of individual characters instead of separated words like Chines. Some of these algorithms include character trigrams and skip-grams as bag-of-character n-grams [2].

#### CNNs

General need / effectiveness
Sentence modelling: 
Window approach: word-based predictions

#### RNNs

Vanilla
LSTM
GRU

word-level, sentence-level, language generation

attention mechanisms - MemNet

Transformer - BERT && OpenAI_GPT

#### Recursive NNs

Constituency-based trees - RNTNs

#### Deep reinforcement learning applications

basic overview

#### Unsupervised && Generative && Memory augmentation

VAEs GANs

#### Bilingual evaluation uderstudy (BLEU)

## 2. Seq2seq with Cross-Entropy & REINFORCE

#### seq2seq

#### Cross-Entropy (log-likelihood)

teacher-forcing vs curriculum learning

#### REINFORCE


## 3. Training

results deteriorating, while training improving -> overfitting to the limited dataset base of dialogues

## 4. Results & Discussion

Telegram

## 5. Future Work


## References

[1] D. Jurafsky and J.H. Martin, "Speech and Language Processing (Unpublished Draft)", 2019.

[2] T. Young, D. Hazarika, S. Poria and E. Cambria, "Recent Trends in Deep Learning Based Natural Language Processing", 2017.

[3] R. Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng and C. Potts, "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", 2013.


