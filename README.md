# Leveraging the power of Deep Reinforcement Learning for training NLP algorithms


**<div style=justify>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This project empirically shows the benefits of combining Deep Reinforcement Learning (DLR) methods with popular Natural Language Processing (NLP) algorithms in the pursuit of state-of-the-art results in dialogue interactions and other human language comprehension tasks. The experiment is based on the simple Cornell University Movie Dialogs database and integrates the sequence-to-sequence (seq2seq) model of LSTM networks into cross-entropy learning for pretraining and into the REINFORCE method. Thus, the algorithm leverages the power of stochasticity  inherent to Policy Gradient (PG) models and directly optimizes the BLEU score, while avoiding getting the agent stuck training from scratch through transfer learning of log-likelihood training. This combination results in improved quality and generalization of NLP models and opens the way for stronger algorithms for various tasks, including those outside the human language domain.**</div>

-------
**1. Preliminaries.** Introduces a conceptual background on the NLP literature and state-of-the-art algorithms for conversational modelling, machine translation and other key challenges in the field; as well as the BLEU metric against which the model will be evaluated.

**2. seq2seq with Cross-Entropy & REINFORCE - the algorithms.** Details the specifics of the algorithms used for this particular experiment and the core structure of the approximation models employed.

**3. Training.** Analyses the progress, duration and statistics of the two different training methods until halting.

**4. Results & Discussion.** Tests the chatbot agent generated from the model in the free open Telegram environment.

**5. Future work.** Explores potential avenues of interest for future experiments.


---------
# 1. Preliminaries


# 2. Seq2seq with Cross-Entropy & REINFORCE


# 3. Training


# 4. Results & Discussion

results deteriorating, while training improving -> overfitting to the limited dataset base of dialogues

# 5. Future Work
