# Leveraging the power of Deep Reinforcement Learning training NLP algorithms


<p align=justify><b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue systems and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.</b>

-------
<p align=justify><b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#1-preliminaries>1. Preliminaries.</a></b> Introduces a conceptual background on the NLP literature and state-of-the-art algorithms for conversational modelling, machine translation and other key challenges in the field.

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#2-seq2seq-with-cross-entropy--reinforce>2. seq2seq with Cross-Entropy & REINFORCE - the algorithms.</a></b> Details the specifics of the algorithms used for this particular experiment and the core structure of the approximation models employed.

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#3-training>3. Training & Tests Discussion.</a></b> Analyzes the progress of the two different training methods until halting, and the corresponding performance of the model on the tests.

<b><a href=https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL#5-future-work>4. Future work.</a></b> Explores potential avenues of interest for future experiments.</p>


---------
## 1. Preliminaries

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The recent advancements on deep neural network architectures, sustained on improved computational capabilities and larger, more comprehensive datasets, have propelled a vast amount of success in the field of Machine Learning. This, coupled with better systems for vectorial representation of language structures in the form of embeddings, has put Natural Language Processing at the forefront of research and progress. The following subsections serve as an overview of major methods for different NLP tasks and the works that led to this implementation of seq2seq based on Ranzato et al. [6].</p>

#### Embeddings

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Embeddings are distributional vectors representing different levels of linguistic structures (characters and words). They capture meaning by encoding reference attributes to each structure based on the context in which it appears, i.e. the other words and characters that tend to be next to the target structure [1-Ch6].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>Word2vec</b> are some of the most popular word embedding algorithms. The <b>skip-gram</b> model predicts context words based on the target word. It trains a logistic regression classifier that computes the conditional probability between pairs of words with the dot-product between their embeddings. Opposite to skip-gram, the continuous bag-of-words (<b>CBOW</b>) predicts a target word from the context words. Based on fully-connected NNs with one hidden-layer, these methods allow for efficient representations of dense vectors that capture semantic and syntactic information. However, they are weak on sentiment, polisemy, phrase meaning, conceptual meaning and out-of-vocabulary (OOV) words, which is problematic for some tasks [2, 1-Ch6].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Contextualized word embeddings are another type of embeddings that directly address some of these issues. <b>ELMo</b> creates a different word embedding for each context in which a word appears, thus capturing polisemic meaning. It consists of a bidirectional language model with a forward Long Short-Term Memory (LSTM) network that calculates the joint probability of a series of input tokens and predicts the next token, a backward LSTM network that predicts the previous token and the cross-entropy loss between the two predictions.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Alternatively, approaching the embedding problem at the character-level has allowed researchers to tackle some issues aforementioned and tasks like named-equity recognition (NER), adding meaning to phrases by representing words simply as a combination of characters. Additionally, they prove more effective with some morphologically-rich languages like Spanish, and languages where text is composed of individual characters instead of separated words like Chinese. Some of these algorithms include character trigrams and skip-grams as bag-of-character n-grams [2].</p>

#### RNNs

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The most popular neural network architectures are recurrent neural networks (<b>RNNs</b>), since they are able to capture the sequential nature of language through the transfer of a latent state called the hidden state to the next input in the network. Thus, each input becomes dependent on the sequence of previous inputs, in addition to being dependent on itself.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The most basic model is the <b>vanilla RNN</b>. The network takes as input the hidden state from the previous input state, <i>h-1</i> and the current input state, <i>x</i>. To calculate the current hidden state, it first multiplies <i>h-1</i> and <i>x</i> by two weight matrices and then applies a non-linearity to the sum of both. The current hidden state is passed onto the next RNN in the process and the output of the network is subsequently calculated as a non-linearity of the current hidden state times a weight matrix. This individual RNN processes can be put together following different model architectures, such as taking one RNN's process output as the next RNN state input <i>x</i>, and many more.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vanilla RNNs suffer from the exploding/vanishing gradients problem. <b>LSTM</b> networks were specifically designed to address this issue. They consist of forget, input and output gates. LSTM calculates the cell state and the hidden state as functions of these gates. The three different gates are calculated as a sigmoid of weights matrices times the concatenated <i>h-1 & x</i> plus a bias. The forget gate decides what to forget from the previous cell state, the input gate decides which inputs will be updated and the output gate helps to calculate the next hidden state. Finally, the current cell state is calculated as the dot-product between the forget gate and the previous cell state plus the dot-product between the input gate and a non-linearity of a weight matrix times the concatenated <i>h-1 & x</i> plus a bias. The latter terms regulates the network. Finally, the next hidden state is caculated as a dot-product of the output gate and a non-linearity of the current cell state.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Gated recurrent units (<b>GRUs</b>) are another proposal which follows the same philosophy as LSTMs but are computationally cheaper and faster, since they perform fewer tensor operations. It gets rid of the cell state memory and includes two gates: the update gate and reset gate. The reset gate decides how much past information to forget and the update gate how much of the previous hidden state to do away with and how much of the current network state to add. Both LSTMs and GRUs have proven to be superior in quality to vanilla RNNs, but do not show huge differences in outcome among them.</p>

#### CNNs

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Convolutional neural networks (<b>CNNs</b>) are very effective feature abstraction tools that can extract high-level information from large corpora and their embedded representations. CNNs have been used to create latent semantic representations of sentences, obtaining a global summarization of the sentence features through deep layers of convolutions [2].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Often times, however, word-level representations are required for many NLP tasks and RNNs have traditionally been prioritized since they are designed to capture the sequential nature of language while CNNs draw a broader generalized overall picture. Some recent works have been able to address sequence modelling through the <b>window approach</b>. Convolutions are applied to a window of words of size <i>k</i> around the target word. Thus, the CNN is able to extract contextual meaning for words from its neighbors.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Especially relevant has been the success of Gated CNNs (<b>GCNNs</b>), which have beaten previous state-of-the-art results from LSTM-based recurrent models at some NLP tasks. This architecture consists of encoder-decoder convolution models and is constructed upon 4 solid foundations. Firstly, it employs gated linear units (<b>GLUs</b>) non-linearities that allow the networks to change the scope of abstraction from the full input field to fewer elements within it by covenience. Secondly, for decoder networks it <b>caps the convolution window</b> at the front so it will not learn to make word predictions having already considered future information. Thirdly, it uses <b>residual connections</b> from the input to layer outputs as proposed in <i>ResNet</i> that allow for deeper convolutions. Finally, it employs a <b>multi-step attention mechanism</b> which informs the decoder about the full history of previous inputs having been considered, while RNNs may partially lose this sequence information as it travels through multiple non-linearities [4, 5].</p>

#### Recursive NNs

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Sequential processing of sentences has yielded some positive results. However, it sometimes fails to capture the compositionality aspect of languages. Longer sentences can underperform in previous models, especially in tasks like sentiment analysis, because they do not take into account that words group into phrases before reaching the higher structural level of sentences.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Recursive neural networks are based on constituency parsing trees and are better at this job. One of the most succesful architectures so far has been the recurrent neural tensor network (<b>RNTN</b>). Words are represented as vectors and leaf nodes within the tree structure. The model performs matrix and tensor computations to move from the leaf nodes upwards, as can be seen in Figure 1. Here, the <i>f</i> function represents a non-linearity and the indexes of the <i>V</i> tensor the dimensionality of the tensor. When the model reaches the root node, it performs a matrix multiplication with a set of weights and finally applies a softmax activation function. Thus, it is able to capture sentence sentiment with extraordinarily high  levels of accuracy [3].</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/RNTNtree.png" height=255 width=446><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/RNTN.png" height=255 width=300></p>
<p align="center"><b>Figure 1:</b> RNTN. <b>1 left</b> RNTN tree. <b>1 right</b> RNTN matrix operations. <b>Source:</b>Own elaboration and Socher et al. [7].</p>

#### Generative

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;DeepFakes have put deep generative models at the center of controversy and discussion, especially with fake videos about high-ranking officials. There has been some research in the last couple of years seeking to apply these models in NLP tasks, revolving around the two most common structures: VAEs and GANs.</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Variational autoencoders (<b>VAEs</b>) consist of an encoder-decoder scheme of NNs whichs is iteratively optimized. Unlike vanilla autoencoders, VAEs regularize the latent space according to a Gaussian normal distribution in order to obtain a stable and reliable representation of language structures so that the generator can sample new meaningful states from it. The loss function is calculated as the sum of the reconstruction error loss (the mean-squared error between the encoded input and decoded output) and the regularisation loss (the Kullback-Leibler divergence between the encoded distribution and the normal Gaussian distribution) [9]. Some modifications have successfully been implemented, like augmenting the unstructured <i>z</i> space with structured <i>c</i> variables which target specific salient semantic features of sentences (e.g. a positive/negative attribute for sentiment) [10].</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/VAE.png" height=305 width=839></p>
<p align="center"><b>Figure 2:</b> VAE generation diagram. <b>Source:</b> Rocca [9].</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/VAEs.png" height=305 width=645></p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Generative adversarial networks (<b>GANs</b>) are systems consisting of a neural network generator and discrimator. The generator will be fed random noise, from which it will create new data which replicates as much as possible the underlying distribution of some real data. The job of the generator is to trick the discriminator into believing that its output comes from real data (discriminator accuracy of 0.5), while the discriminator's mission is to separate fake from real inputs [11]. In 2017 a GAN adaptation by Yang et al. [12] achieved state-of-the-art results on machine translation by constructing a GAN with a transformer generator, a CNN discriminator and BLEU reinforced learning called BR-CSGAN.

#### Attention Mechanisms

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In encoder-decoder systems sometimes encoders are inefficiently forced to encode embeddings that are not fully relevant. Attention mechanisms bound decoders by a history of the input data in addition to the previous latent state and generated token. This works as a mapping between certain value pairs and allows the network to focus on specific data from the whole dataset, essentially adding context at different decoding timesteps [1-Ch10, 2].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;End-to-end memory networks (<b>MemNets</b>) adopt this approach in a way that the attention system resembles a sort of internal memory. The model stores all the embedded input sentences in a memory and embeds the query as well. Importance weights of the memory input items are calculated by taking a softmax of the dot-product between the embedded query and the input memory. These weights represent the attention or importance given to each input data. The importance-adjusted input data is then added to the embedded query, and processed through a weight matrix and a softmax to generate the final prediction. This process can be performed in an iterative way as shown in Figure 2 for more clarity [7].</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/MemNet.png" height=305 width=675></p>
<p align="center"><b>Figure 2:</b> MemNet. <b>2a</b> Single-layer MemNet. <b>2b</b> Multi-layer MemNet. <b>Source:</b> Sukhbaatar et al. [7].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Another landmark architecture employing attention mechanisms is the <b>transformer</b> network. This model replaces the RNNs and CNNs typically used in encoder-decoder frameworks with attention layers, as shown in Figure 3. The encoder consists of a series of identical stacked layers, each with two sublayers: a <b>multi-head</b> attention mechanism and a normal fully-connected network. The decoder is similar to the encoder, but it includes an extra multi-head attention layer to process encoder output. Additionally, information from future positions in the input to the first multi-head layer is masked, since it would be cheating to make predictions with input from steps ahead. The model uses residual connections in each sublayer, followed by layer normalization.</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/transformer.png" height=450 width=335></p>
<p align="center"><b>Figure 3:</b> Transformer network. <b>Source:</b> Vaswani et al. [8].</p>


<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Multi-head attention layers perform parallel dot-product attention functions on the queries, keys and values, which have been previously linearly projected <i>h</i> times. The outputs are then concatenated and projected linearly again. Thus, the model is able to learn from different representation subspaces at different positions [8].</p>

<p align="center"><img src="https://github.com/inigo-irigaray/NLP-seq2seq-with-DeepRL/blob/master/images/multihead.png" height=305 width=675></p>
<p align="center"><b>Figure 4:</b> Multi-head attention function. <b>4 left</b> Dot-product attention. <b>4 right</b> Multi-head attention. <b>Source:</b> Vaswani et al. [8].</p>

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Building upon the <i>transformer</i> model, Open AI released the generative pre-training model (<b>GPT</b>). The training process consisted of 2 stages, unsupervised pretraining of the transformer algorithm, followed by a supervised fine-tuning, task-specific process. The former trains the transformer for language modelling on a corpus of unsupervised tokens by maximizing the log loss of conditional probabilities of the current token and a context window of past token sequences. The latter assumes a labelled set of data to pass through the transformer architecture, after which a linear transformation layer is added with a softmax activation function for prediction. It then maximizes the log loss between predicted and actual inputs plus a language modelling loss similar to the one used for unsupervised training [13].

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>BERT</b>

#### Deep reinforcement learning applications

<p align=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;With the advancement of deep reinforcement learning some new training techniques have been proposed, where the generator networks are considered as the agent and the objective metric as the reward to be maximized [2]. One of these methods is the one proposed by Ranzato et al. [6] which is the basis for this experiment and will be explained in depth in the next section.

## 2. Seq2seq with Cross-Entropy & REINFORCE

#### seq2seq

basic black box encoder decoder

#### Cross-Entropy (log-likelihood)

basics
teacher-forcing
curriculum learning
alternation

#### REINFORCE

basics of REINFORCE
adaptation to NLP

## 3. Training & Results

#### Bilingual evaluation uderstudy (BLEU)

general description and analysis

27% improvement. results deteriorating, while training improving -> overfitting to the limited dataset base of dialogues

## 4. Future Work


## References

#### <p>[1] D. Jurafsky and J.H. Martin, "Speech and Language Processing (Unpublished Draft)", 2019.</p>

#### <p>[2] T. Young, D. Hazarika, S. Poria and E. Cambria, "Recent Trends in Deep Learning Based Natural Language Processing", 2017.</p>

#### <p>[3] R. Socher, A. Perelygin, J.Y. Wu, J. Chuang, C.D. Manning, A.Y. Ng and C. Potts, "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", 2013.</p>

#### <p>[4] Y.N. Dauphin, A. Fan, M. Auli and D. Grangier, "Language Modeling with Gated Convolutional Networks", 2016.</p>

#### <p>[5] J. Gehring, M. Auli, D. Grangier, D. Yarats and Y.N. Dauphin, "Convolutional Sequence to Sequence Learning", 2017.</p>

#### <p>[6] M. Ranzato, S. Chopra, M. Auli and W. Zaremba, "Sequence Level Training with Recurrent Neural Networks", 2015.</p>

#### <p>[7] S. Sukhbaatar, A. Szlam, J. Weston and R. Fergus, "End-to-end Memory Networks", 2015.</p>

#### <p>[8] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A.N. Gomez and L. Kaiser, "Attention is All You Need", 2017.</p>

#### <p>[9] J. Rocca, "Understanding Variational Autoencoders (VAEs)", <i>Towards Data Science</i>, 2019.</p>

#### <p>[10] Z. Hu, Z. Yang, X. Liang, R. Salakhutdinov and E.P. Xing, "Toward Controlled Generation of Text", <i>arXiv:1703.00955</i>, 2017.</p>

#### <p>[11] I.J. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville and Y. Bengio, "Generative Adversarial Networks", <i>arXiv:1406.2661</i>, 2014.</p>

#### <p>[12] Z. Yang, W. Chen, F. Wang and B. Xu, "Improving Neural Machine Translation with Conditional Sequence
Generative Adversarial Nets", <i>arXiv:1703.04887</i>, 2017.</p>

#### <p>[13] A. Radford, K. Narasimhan, T. Salimans and I. Sutskever, "Improving Language Understanding by Generative Pre-Training", <i></i>, 2017.</p>
