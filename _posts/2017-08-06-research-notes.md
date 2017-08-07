---
layout: post
title:  "Research Notes"
date:   2017-08-06 21:53:05 -0400
categories: notes
use_math: true
lang: en
---

This post records my notes taken during summer internship @ CMU LTI. 

Disclaimer: These notes are not guaranteed to be correct or understandable.

Note: This is a non-mobile-friendly post, mobile view is distorted due to formulae.

<!--more-->

## About Me

Dept. of Computer Science and Technology, Tsinghua University

Beginning senior year next semester

Here as a visiting intern for the summer program

A beginner in the field of NLP



## Research Topic

#### General Idea

A language model that can predict tokens and phrases. Phrases are extracted from training data. To speed up training, a subset of phrases are selected at each time step. Probabilities are marginalized over all possible segmentations of training sentence to phrases and tokens.

#### Description

Basically, our model can either predict a token or a phrase. Particularly, we want to use all phrases from the training data that are interesting. A nice criteria could be to find maximal substrings with more than one occurrence. Suffix data structures could be utilized to efficiently find the substrings.

Naive model would use a softmax over all tokens and phrases, but this is computationally expensive. Methods we can try:

- Negative sampling & importance sampling
- Hierarchical softmax (Graham said may not work too well)
- KNN search: linearly transform hidden state into embedding space, and find top-K nearest neighbors as candidate phrases. Locality sensitive hashing could be applied, as only hashes of the K phrases would be changed
  This method is also easier for the model to generalize to machine translation tasks

We may also need to manually include true match phrases in addition to top-KNNs during (the beginning of) training, because the model may not be able to find the correct candidates at the beginning.

There may be more than one correct match: any prefix of the current suffix of training sentence is a correct match. We need to marginalize over different paths using either Latent Predictor Networks or Neural Lattice Models.

#### Ideas

- **On combining char-level and word-level embeddings**
  - Word-level embeddings can be represented by running a Bi-LSTM over individual character embeddings, thus reducing model size.
  - Computationally expensive; can be improved by first using above methods (KNN search etc.) to select candidates and then run LSTMs for the selected subset.
- ​

## Current Progress

#### Week 1 (6.29 - 7.5)

**Objective**: Read course materials and implement homework

**Progress**

- Just beginning, learning the basics of language models
- Trained LSTM models using DyNet
- Configured the TIR cluster

#### Week 2 (7.6 - 7.12)

**Objective**: Tune the model to match paper performance; learn enhanced suffix arrays

**Progress**

- Matched paper performance
- Read papers regarding the problem
- Learned about basics of suffix arrays

#### Week 3 (7.13 - 7.19)

**Objective**: Implement LPN or lattice model, using frequent bigrams (or maybe words?) as vocabulary

**Progress**

- Implemented LPN with top-1000 words as vocabulary


#### Week 4 (7.20 - 7.27)

**Objective:** Tune the LPN model to achieve decent perplexity values

**Progress**

- Did experiments using LPN variations
- Refactored code
- Implemented truncated backprop


#### Week 5 (7.28 - 8.4)

**Objective:** Same as last week

**Progress**

- ​

## Tips in Training

#### Dropout

- Apply a random mask on parameters: each parameter is zeroed with a probability of $p$, note that output would be scaled down to $1-p$ compared to values when dropout is not applied
- **Inverted dropout**: Rescale output to $1/(1-p)$ during training, so no special treatment is required for using the models
- **Dropout for embeddings**: zero out entire vectors for random word IDs
- **Dropout for LSTMs**: apply to <u>input</u> and <u>hidden state</u>, rather than parameters. Dropout mask is the same for each time step on one training sample *(ref: [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf))*
- **Dropout for final FC layer in LSTMs**: apply to <u>LSTM output</u>, rather than FC parameters

#### Evaluating similarity

- To evaluate similarities of $h_L$ and $h_R$ based on ratings of $[1,K]$, we can jointly train a MLP based on:
- $$
  \begin{align*}
  h_\times &= h_L\odot h_R \\
  h_+ &= \vert h_L-h_R \vert \\
  h_s &= \sigma\left(W^{(\times)}h_\times + W^{(+)}h_+ + b^{(h)}\right) \\
  \hat{p_\theta} &= \mathrm{softmax}\left(W^{(p)}h_s + b^{(p)}\right) \\
  r &= \{1,2,\ldots,K\} \\
  \hat{y} &= r^\top \hat{p_\theta}
  \end{align*}
  $$

  i.e. <u>learning a evaluation criterion based on distance and angle</u> between the pair, and mapping it as a weighted average of ratings. Obviously the resultant prediction $\hat{y}$ will be in the range $[1,K]$.

- **Triplet loss**: *<u>(more details to be described)</u>*


#### Gumbel-max trick

- **Gumbel distribution** (unit scale, zero location, $x\in(-\infty,+\infty)$):
  - **PDF**: $f(x) = \exp(-x-\exp(-x))$
  - **CDF**: $F(x) = \exp(-\exp(-x))$
  - Property: If $U\sim \mathrm{Uniform}[0,1]$, then $-\log(-\log U)\sim \mathrm{Gumbel}(0,1)$

- Softmax for $\{x_k\}$ is equivalent to: adding independent Gumbel noise and take argmax

- Proof: Let $z_k = x_k + y_k,\ \{y_k\}\stackrel{\mathrm{i.i.d.}}{\sim}\mathrm{Gumbel}(0,1)$, then $P(z_k\text{ is max}) = \prod_{j\neq k} F(z_k-x_j)$.

  $$
  \renewcommand{\d}{\ \mathrm{d}}
  \begin{align*}
  P(k\text{ is selected}) &= \int_{-\infty}^{+\infty}f(z_k-x_k)P(z_k\text{ is max})\d{z_k} \\
  &= \int \exp\left(-z_k+x_k-\exp(-z_k)\sum_{j=1}^{K}\exp(x_k)\right)\d{z_k} \\
  &= \frac{\exp(x_k)}{\sum_{j=1}^{K}\exp(x_j)}=\mathrm{softmax}\left(\{x_k\}\right)^{(k)}
  \end{align*}
  $$

- See also: [https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/](https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/)

#### Gaussian reparameterization trick

- *<u>(more details to be described)</u>*

#### Exposure bias and scheduled sampling

- **Exposure bias**: In sequential models, during training, we feed the ground-truth label at the previous time step as input, no matter what the output prediction was; while during testing, we always feed the previous output. This way, the model is "exposed" to the ground-truth, even when it was not able to make such predictions. The model may also fail to capture relations between the next state and the previous output.
- **Scheduled sampling**: At each time step, use previous label by probability $1-p$, and use "teacher forcing" (feed ground-truth) by probability $p$.
  $p$ is set to a high value at the beginning of training, and eventually anneal to a close to 0 value (thus the name "scheduled" sampling).
  See also: [https://www.evernote.com/shard/s189/sh/c9ac2e3f-a150-4d0c-9a44-16657e5d42cd/5eb49d50695c903ca1b4a04934e63363](https://www.evernote.com/shard/s189/sh/c9ac2e3f-a150-4d0c-9a44-16657e5d42cd/5eb49d50695c903ca1b4a04934e63363)
- **Drawbacks of scheduled sampling**: When previous label was used, we were using the result of argmax of the softmax at previous time step as input, and naturally we would like to back propagate through such calculations. However, argmax is non-differentiable.
  The reason back propagating through argmax is desirable is that, the actual cause for predicting a wrong label at the current time step may be that wrong predictions were made at previous time steps (cascading error).
- **Soft argmax and differentiable scheduled sampling** *(ref: [Differentiable Scheduled Sampling for Credit Assignment](https://arxiv.org/pdf/1704.06970.pdf))*: *<u>(more details to be described)</u>*


#### Truncated Backprop

- Concretely, for every $k_1$ time steps, train on the following $k_2$ time steps. When $k_1<k_2$, there's overlap between consecutive time steps; sometimes $k_1=k_2$ is desired.

  ![](https://r2rt.com/static/images/RNN_tf_truncated_backprop.png)

- Initial state may be **zeroed** by a small probability, so as to bias the model towards being easily start from a zero state in test time *(ref: [On the State of the Art of Evaluation in Neural Language Models]((https://arxiv.org/pdf/1707.05589.pdf)))*

- **Data preprocessing**: *<u>(more details to be described)</u>*

- **Pros**: cheaper to train (less memory consumption for computation graphs), and mitigates the vanishing gradient problem; **Cons**: constrained the maximum range for dependencies


#### Entropy, Cross Entropy Loss, and KL-Divergence

- Shannon **Entropy** of a probability distribution is defined as
  
  $$
  H(p)=\mathbb{E}_p[-\log p]=-\sum_{x_i}p(x_i)\log p(x_i)
  $$

  which is the expected number of bits required to represent an element in the set over which the probability distribution is defined. The lower bound for the number of bits required to represent an element $x_i$ is $\log\frac{1}{p(x_i)}=-\log p(x_i)$.

- **Cross-Entropy loss** is defined on two distributions:

  $$
  H(p,q)=\mathbb{E}_p[-\log q]=-\sum_{x_i}p(x_i)\log q(x_i)
  $$

  which can be interpreted as estimating entropy using the wrong probability $q$. When minimizing w.r.t. cross-entropy loss, we're trying to match our predicted distribution $q$ to the true distribution $p$.

- **KL-divergence** is simply the difference between entropy and cross-entropy loss:

  $$
  \mathrm{KL}(p\ \Vert\  q)=H(p,q)-H(p)=\sum_{x_i}p(x_i)\log\frac{p(x_i)}{q(x_i)}
  $$

  which is the number of extra bits required. Usually minimizing w.r.t. KL-divergence is equivalent to minimizing w.r.t. cross-entropy loss.

- See also: [https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/) and [https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)

#### Tied Input and Output Embeddings

- Let $L$ be the input embedding, such that input to the LSTM is $x_t = Ly^*_{t-1}$
- Replace the dense layer $y_t=\mathrm{softmax}(Wh_t+b)$ following the LSTM unit by the **transpose** of the embedding, i.e. $y_t=\mathrm{softmax}\left(L^\top h_t\right)$ *(ref: [Tying Word Vectors and Word Classifiers etc.](https://arxiv.org/pdf/1611.01462.pdf))*
- Since input and output are in the same space, it is reasonable to assume they're related by a linear transformation $A$. Tying embeddings results in minimizing w.r.t. vector similarities:
  - Let $u_t=Ly^*_t$, i.e. the embedding of the actual output. By minimizing w.r.t. vector similarities, we would like the probability $y_t$ be related to similarity metrics, concretely $y_t=\tilde{y_t}=\mathrm{softmax}\left(L^\top u_t\right)$, where we use inner product as measurement of similarity.
  - In order to minimize loss, $h_t$ would be adjusted to be closer to the appropriate column of $L$.
  - If we apply KL-divergence as loss, $\tilde{y_t}$ could be used as the estimated true distribution. Other class labels are also utilized during backprop, compared to the case when one-hot encoding is used.

#### Softmax Approximations by Sampling

- Ref to: [http://ruder.io/word-embeddings-softmax/index.html#samplingbasedapproaches](http://ruder.io/word-embeddings-softmax/index.html#samplingbasedapproaches)

- Given the correct word $w$, and all candidate words $w_i$. For negative log softmax loss, the formula for loss is:

  $$
  J_w=-\log\frac{\exp(h^\top v_{w})}{\sum_{w_i}\exp(h^\top v_{w_i})}=-h^\top v_w+\log\sum_{w_i}\exp(h^\top v_{w_i})
  $$

  where $v_w$ is the output embedding. Denoting $\mathcal{E}(w)=-h^\top v_{w}$, taking gradients w.r.t. parameters would give:

  $$
  \begin{align*}
  \nabla_\theta J_w & = \nabla_\theta \mathcal{E}(w)+\nabla_\theta \log\sum_{w_i}\exp(-\mathcal{E}(w_i)) \\
  & = \nabla_\theta\mathcal{E}(w)-\sum_{w_i}\frac{\exp(-\mathcal{E}(w_i))}{\sum_{w_i'}\exp(-\mathcal{E}(w_i'))}\nabla_\theta\mathcal{E}(w_i) \\
  & = \nabla_\theta\mathcal{E}(w)-\sum_{w_i}P(w_i)\nabla_\theta\mathcal{E}(w_i) \\
  & = \nabla_\theta\mathcal{E}(w) - \mathbb{E}_{w_i\sim P}[\nabla_\theta\mathcal{E}(w_i)]
  \end{align*}
  $$

  where $P(w_i)$ is the softmax probability of $w_i$.

  Sampling methods reduce computational complexity by approximating the expected term.

- **Importance sampling**: 

  - Expectation can be calculated using Monte Carlo methods: average of samples.
  - To avoid computing actual probabilities (which is the same as calculating softmax), use another distribution $Q$ similar to the target distribution $P$.

- **Noise contrastive estimation** (NCE):

  - Ref to: [https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss](https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss) and [https://arxiv.org/pdf/1410.8251.pdf](https://arxiv.org/pdf/1410.8251.pdf)

  - Language modeling can be seen as a multinomial classification problem (predicting the label of the next word). We can convert this into a binary classification problem.

  - Train the LM as is but w/o the final output layer. Jointly train an extra binary classifier to distinguish noise (randomly chosen words) against correct words $w$ given the context $c$.

  - For each word, sample $k$ noises $\tilde{w}_{k}$ from noise distribution $Q$. Minimize per-category cross-entropy loss using logistic regression, giving the loss function, and substituting expectation with Monte Carlo sampling:

    $$
    \begin{align*}
    J_w & \stackrel{\phantom{M.C.}}{=} - \log P(y=1\mid w,c) - k\cdot\mathbb{E}_{\tilde{w}_{j}\sim Q}[\log P(y=0\mid \tilde{w}_{j},c)] \\
    & \stackrel{M.C.}{=} - \log P(y=1\mid w,c) - \sum_{j=1}^{k}\log P(y=0\mid \tilde{w}_{j},c)
    \end{align*}
    $$

    the reason why we used expectation for the noise entropy but not for the positive entropy, is because when we sum over all training data, the positive part would be equal to the entropy calculated for the whole dataset (distribution).

  - So samples come from a mixture of two distributions: the actual empirical distribution $\tilde{P}$ from data (the distribution we're trying to model), and the noise distribution $Q$. We replace the empirical distribution with the learned distribution $P_\theta$ of our model, which gives:

    $$
    \begin{align*}
    P(w\mid c) & = P(y=0,w\mid c)+P(y=1,w\mid c) \\
     & =\frac{k}{k+1}Q(w)+\frac{1}{k+1}P_\theta(w\mid c) \\
     P(y=1\mid w,c) & = \frac{P(y=1,w\mid c)}{P(w\mid c)}=\frac{P_\theta(w\mid c)}{P_\theta(w\mid c)+k\cdot Q(w)} \\
     P(y=0\mid w,c) & = \frac{k\cdot Q(w)}{P_\theta(w\mid c)+k\cdot Q(w)}
    \end{align*}
    $$

  - Substituting probabilities into the loss function, we can calculate its gradients as follows:

    $$
    \begin{align*}
    \nabla J_w & = -\nabla\log P(y=1\mid w,c) - k\cdot\mathbb{E}_{w_j\sim Q}[\nabla\log P(y=0\mid w_j,c)] \\
     & = -\nabla\log P(y=1\mid w,c) - \sum_{w_j\in V}k\cdot Q(w_j)\nabla\log P(y=0\mid w_j,c) \\
     & = -\frac{k\cdot Q(w)}{P_\theta(w\mid c)+k\cdot Q(w)}\cdot\nabla\log P_\theta(w\mid c)+\sum_{w_j\in V}\frac{k\cdot Q(w_j)}{P_\theta(w_j\mid c)+k\cdot Q(w_j)}\nabla P_\theta(w_j\mid c) \\
     & = -\sum_{w_j\in V}\frac{k\cdot Q(w_j)}{P_\theta(w_j\mid c)+k\cdot Q(w_j)}\left(\tilde{P}(w_j\mid c)-P_\theta(w_j\mid c)\right)\nabla\log P_\theta(w_j\mid c)
    \end{align*}
    $$

    where the empirical distribution $\tilde{P}(w_j\mid c)$ equals 1 iff $w_j=w$.

    We can observe that when $k\rightarrow\infty$, the gradient $\nabla J_w\rightarrow -\sum\left(\tilde{P}(w_j\mid c)-P_\theta(w_j\mid c)\right)\nabla\log P_\theta(w_j\mid c)$, which goes to zero as $P_\theta$ matches $\tilde{P}$.

  - But $P_\theta(w\mid c)=\mathrm{softmax}(h^\top v_w)$, which is what we need to estimate. We can replace it by $P_\theta(w\mid c)=\exp(h^\top v_w)/Z(c)$, where $Z(c)$ is trainable. Or simply, let $Z(c)\equiv 1$, giving $P_\theta(w\mid c)=\exp(h^\top v_w)$.

  - **Note**: Performance is poor.

- **Negative sampling**: 

  - An approximation to NCE, by setting the most expensive term $k\cdot Q(w)\equiv1$, giving:

    $$
    P(y=1\mid w,c)=\frac{\exp(h^\top v_w)}{\exp(h^\top v_w)+1}=\frac{1}{1+\exp(-h^\top v_w)}=\sigma(h^\top v_w)
    $$

    where $\sigma$ is the sigmoid function.

  - Equivalent to NCE only when $k=\lvert V\rvert$ and $Q$ is a uniform distribution.

  - **Note**: Inappropriate for language modeling, because probabilistic information is lost. Good for representation learning, as in word2vec.




## Paper Reading

#### [Latent predictor networks](https://arxiv.org/pdf/1603.06744.pdf) by W. Ling et al

- When using a normal RNN model, after inference is made, we can backtrack from the final state through to the initial state, resulting in a path
- This is due to that RNNs generated one token at a time, and at each time step samples the next token according to calculated probabilities
- If RNNs can generate multiple tokens in one time step, there may be <u>multiple paths from the initial state to the target state, corresponding to segmentations of the sequence</u>. Path counts can be exponential to its length, and the union of the paths is a directed acyclic graph
- Paper proposes a method to perform <u>joint training on several predictors of different granularity</u>. The method introduced latent variables for deciding which predictor to use, thus giving it the name
- To calculate gradients for a time step, <u>summed products of probabilities on the DAG</u> are required, which can be calculated using a dynamic programming algorithm
- Attention over all different fields in a structured input is used
- Prediction is done using beam search
- The authors utilized this technique in code generation tasks for card games, where a character-level LSTM predictor is jointly trained with pointer networks for copying text directly from card descriptions
- **Relation to project**: If we were to combine results from suffix DS with word- or character-level predictors, we would need such methods to choose which predictor to use

#### Neural lattice language models by Jacob (Graham's grad. student)

- Main idea is similar: enabling LSTM models to <u>generate multiple tokens in one time step</u>
- Exact probability is hard to calculate as LSTMs keep track of whole context seen from the initial state, so each path would have a different state
- Paper evaluated different approaches of probability estimations and different representations of multiple-tokens in one time step *<u>(more details to be described)</u>*
  - Ancestral sampling from "latent sequence decompositions": just treat multiword tokens as regular tokens
  - TreeLSTM-style summation: summing predecessors' hidden states. Cons: losses probabilistic info
  - Weighted expectation: weight summations using prob. dist. learned in ancestral sampling
- **Difference with "<u>latent predictor networks</u>"**:
  - Latent predictor networks combine multiple predictor models, while this is one unified model
  - The reason why probabilities are easy to calculate in said paper is due to the fact that, although predictors of different granularity are used, <u>all predicted tokens are in the same space, and multiple tokens are fed into the character-level network one-by-one</u>. Hidden states of the char-level network is used in the pointer network in turn. So only O(length) states are required in total
- ~~**<u>An idea:</u>** Can we calculate exact probabilities using method similar to that of latent predictor networks? Possible if all transforms are linear, may be feasible if using Taylor series to approximate non-linearities~~

#### [Pointer Networks](http://papers.nips.cc/paper/5866-pointer-networks.pdf) by O. Vinyals et al

- Output is the set of tokens from input, instead of fixed vocabulary
- Basically a seq2seq model with attention, but use attention weights directly as probability from predicting each input token
- Can be trained to select ordered subsets from input, even accomplish difficult tasks as convex hulls, Delauney triangulation and TSP
- See also: [http://fastml.com/introduction-to-pointer-networks/](http://fastml.com/introduction-to-pointer-networks/)

#### [TreeLSTMs](https://arxiv.org/pdf/1503.00075.pdf) by K. S. Tai, R. Socher, and Christopher D. Manning

- A natural generalization of LSTM to tree structures
- Sum children hidden states as $\tilde{h}$, and replace this as $h$ in formulas for normal LSTMs
- Forget gate is different for each child: use only the hidden state of child to calculate forget gate parameters
- Cell state of parent is as usual, summing over cell states of each child with respective forget gates
- Ordered children version exists: use different parameters for each child (depending on its index). Such model has a limit on the maximum branch factor
- **Benefits**: Can make use of sentence structures generated by parsers; better at preserving state, i.e. can cope better with long distance dependencies (since path lengths are shorter on trees)
