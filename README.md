# Leveraging the power of Deep Reinforcement Learning training NLP algorithms


<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue systems and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.</b>

-------
<p align=justify><b>1. Preliminaries.</b> Introduces a conceptual background on the NLP literature and state-of-the-art algorithms for conversational modelling, machine translation and other key challenges in the field; as well as the BLEU metric against which the model will be evaluated.

<b>2. seq2seq with Cross-Entropy & REINFORCE - the algorithms.</b> Details the specifics of the algorithms used for this particular experiment and the core structure of the approximation models employed.

<b>3. Training.</b> Analyzes the progress, duration and statistics of the two different training methods until halting.

<b>4. Results & Discussion.</b> Tests the chatbot agent generated from the model in the free open Telegram environment.

<b>5. Future work.</b> Explores potential avenues of interest for future experiments.</p>


---------
## 1. Preliminaries

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The recent advancements on deep neural network architectures, sustained on improved computational capabilities and larger, more comprehensive datasets, has propelled an enormous amount of success in the field of Machine Learning. This, coupled with better systems for vectorial representation of language structures in the form of embeddings, has put Natural Language Processing at the forefront of research and progress. The following subsections serve as an overview of major methods for different NLP tasks and the works that led to this implementation of seq2seq based on Ranzato et al.</p>

#### Embeddings

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Embeddings are distributional vectors representing different levels of linguistic structures (characters and words). They capture meaning by encoding reference attributes to each structure based on the context in which it apperas, i.e. the other words and characters that tend to appear next to the target structure [1-Ch6].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Word2vec</b> are some of the most popular word embedding algorithms. The <b>skip-gram</b> model predicts context words based on the target word. It trains a logistic regression classifier that computes the conditional probability between pairs of words with the dot product between their embeddings. Opposite to skip-gram, the continuous bag-of-words (<b>CBOW</b>) predicts a target word from the context words. Based on fully-connected NNs with one hidden-layer, these methods allow for efficient representations of dense vectors that capture semantic and syntactic information. However, they are weak on sentiment, polisemy, phrase meaning, conceptual meaning and out-of-vocabulary (OOV) words, which is problematic for some tasks [2, 1-Ch6].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contextualized word embeddings are another type of embeddings that directly address some of these issues. <b>ELMo</b> creates a different word embedding for each context in which a word appears, thus capturing polisemic meaning. It consists of a bidirectional language model of a forward Long Short-Term Memory (LSTM) network to model the joint probability of a series of input tokens and predict the next token, a backward LSTM network that predicts the previous token and the cross-entropy loss between the two predictions. <b>OpenAI-GPT-2, BERT</b></p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Alternatively, approaching the embedding problem at the character-level has allowed to researchers to tackle some issues aforementioned, like OOVs, and tasks like named-equity recognition (NER), adding meaning to phrases by representing words simply as a combination of characters. Additionally, they prove more effective with some morphologically-rich languages like Spanish, and languages where text is composed of individual characters instead of separated words like Chinese. Some of these algorithms include character trigrams and skip-grams as bag-of-character n-grams [2].</p>

#### RNNs

Vanilla
LSTM
GRU

word-level, sentence-level, language generation

attention mechanisms - MemNet

Transformer - BERT && OpenAI_GPT

#### CNNs

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Convolutional neural networks (<b>CNNs</b>) are very effective feature abstraction tools that can extract high-level information from large corpora and their embedded representations. CNNs have been used to create latent semantic representations of sentences, obtaining a global summarization of the sentence features through deep layers of convolutions [2].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Often times, however, word-level representations are required for many NLP tasks and RNNs have traditionally been prioritized since they are designed to capture the sequential nature of language while CNNs draw a broader generalized overall picture. Some recent works have been able to address sequence modelling through the <b>window approach</b>. Convolutions are applied to a window of words of size <i>k</i> around the target word. Thus, the CNN is able to extract contextual meaning for words from its neighbors.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Especially relevant has been the success of Gated CNNs (<b>GCNNs</b>), which have beaten previous state-of-the-art results from LSTM-based recurrent models at some NLP tasks. This architecture consists of encoder-decoder convolution models and is constructed upon 4 solid foundations. Firstly, it employs gated linear units (<b>GLUs</b>) non-linearities that allow the networks to change the scope of abstraction from the full input field to fewer elements within it by covenience. Secondly, for decoder networks it <b>caps the convolution window</b> at the front so it will not learn to make word predictions having already considered future information. Thirsly, it uses <b>residual connections</b> from the input to network outputs as proposed in <i>ResNet</i> that allow for deeper concolutions. Finally, it employs a <b>multi-step attention mechanism</b> which informs the decoder about the full history of previous inputs having been considered, while RNNs may partially lose this sequence information as it travels through multiple non-linearities.</p>

#### Recursive NNs

Constituency-based trees - RNTNs

#### Unsupervised && Generative && Memory augmentation

VAEs GANs

#### Deep reinforcement learning applications

basic overview

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

[4] Y.N. Dauphin, A. Fan, M. Auli and D. Grangier, "Language Modeling with Gated Convolutional Networks", 2016.

[5] J. Gehring, M. Auli, D. Grangier, D. Yarats and Y.N. Dauphin, "Convolutional Sequence to Sequence Learning", 2017.
