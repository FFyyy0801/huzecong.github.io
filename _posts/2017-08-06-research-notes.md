---
layout: post
title:  "Research Notes"
date:   2017-08-06 21:53:05 -0400
categories: notes
mathjax: true
locale: en
---

This post records my notes taken during summer internship @ CMU LTI. 

Disclaimer: These notes are not guaranteed to be correct or understandable.

Note: This is a non-mobile-friendly post, mobile view is distorted due to formulae.

<!--more-->

$$
\renewcommand{\d}{\ \mathrm{d}}
$$

## Tips in Training

#### Dropout

- Apply a random mask on parameters: each parameter is zeroed with a probability of $p$, note that output would be scaled down to $1-p$ compared to values when dropout is not applied

- **Inverted dropout**: Rescale output to $1/(1-p)$ during training, so no special treatment is required for using the models

- **Dropout for embeddings**: zero out entire vectors for random word IDs

- **Dropout for LSTMs**: apply to <u>input</u> and <u>hidden state</u>, rather than parameters. Dropout mask is the same for each time step on one training sample *(ref: [[Gal 2016] A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/pdf/1512.05287.pdf))*

- **Dropout for final FC layer in LSTMs**: apply to <u>LSTM output</u>, rather than FC parameters

#### Evaluating similarity

- To evaluate similarities of $h_L$ and $h_R$ based on ratings of $[1,K]$, we can jointly train a MLP based on:

  $$
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


#### Gumbel-max trick & Gumbel-Softmax distribution

- **Gumbel distribution** (unit scale, zero location, $x\in(-\infty,+\infty)$):

  - **PDF**: $f(x) = \exp(-x-\exp(-x))$
  - **CDF**: $F(x) = \exp(-\exp(-x))$
  - Property: If $U\sim \mathrm{Uniform}[0,1]$, then $-\log(-\log U)\sim \mathrm{Gumbel}(0,1)$

- Softmax for $\{x_k\}$ is equivalent to: adding independent Gumbel noise and take argmax

- **Proof**: Let $z_k = x_k + y_k,\ \{y_k\}\stackrel{\mathrm{i.i.d.}}{\sim}\mathrm{Gumbel}(0,1)$, then $P(z_k\text{ is max}) = \prod_{j\neq k} F(z_k-x_j)$.
  $$
  \begin{align*}
  P(k\text{ is selected}) &= \int_{-\infty}^{+\infty}f(z_k-x_k)P(z_k\text{ is max})\d{z_k} \\
  &= \int_{-\infty}^{+\infty} \exp\left(-z_k+x_k-\exp(-z_k)\sum_{j=1}^{K}\exp(x_k)\right)\d{z_k} \\
  &\stackrel{\text{magic}}{=} \frac{\exp(x_k)}{\sum_{j=1}^{K}\exp(x_j)}=\mathrm{softmax}\left(\{x_k\}\right)^{(k)}
  \end{align*}
  $$
  where the "magic" step somehow calculates the closed-form solution of the above integration.

- See also: <https://hips.seas.harvard.edu/blog/2013/04/06/the-gumbel-max-trick-for-discrete-distributions/>

- **Usage**: Replace sampling from distribution $P(x)=\pi_x$ with argmax operation:
  $$
  z=\underset{i}{\arg\max}(g_i+\log\pi_i)\sim P
  $$
  where $g_i$ are independent samples from uniform Gumbel distribution.

- **Gumbel-Softmax distribution**: Softmax with temperature $\tau$ applied over Gumbel-max:
  $$
  \mathbf{y}=\mathrm{softmax}((\log\pmb{\pi}+\mathbf{g})/\tau)
  $$
  For lower temperatures, Gumbel-Softmax distribution is close to the one-hot distribution of the argmax element (which is the sample given by Gumbel-max trick); for higher temperatures, distribution is close to uniform. *(ref: [[Jang et al. 2016] Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144))*

  - This is useful when we need a differentiable sample over a discrete distribution. For sample over continuous distributions, we have the [reparameterization trick](#how-do-we-compute-the-lower-bound).

#### Exposure bias and scheduled sampling

- **Exposure bias**: In sequential models, during training, we feed the ground-truth label at the previous time step as input, no matter what the output prediction was; while during testing, we always feed the previous output. This way, the model is "exposed" to the ground-truth, even when it was not able to make such predictions. The model may also fail to capture relations between the next state and the previous output.

- **Scheduled sampling**: At each time step, use previous label by probability $1-p$, and use "teacher forcing" (feed ground-truth) by probability $p$.

  $p$ is set to a high value at the beginning of training, and eventually anneal to a close to 0 value (thus the name "scheduled" sampling).
  See also: <https://www.evernote.com/shard/s189/sh/c9ac2e3f-a150-4d0c-9a44-16657e5d42cd/5eb49d50695c903ca1b4a04934e63363>

- **Drawbacks of scheduled sampling**: When previous label was used, we were using the result of argmax of the softmax at previous time step as input, and naturally we would like to back propagate through such calculations. However, argmax is non-differentiable.

  The reason back propagating through argmax is desirable is that, the actual cause for predicting a wrong label at the current time step may be that wrong predictions were made at previous time steps (cascading error).

- **Other caveats**: see [A Word of Caution on Scheduled Sampling for Training RNNs](http://www.inference.vc/scheduled-sampling-for-rnns-scoring-rule-interpretation/).

- **Soft argmax and differentiable scheduled sampling** *(ref: [[Goyal, Dyer 2017] Differentiable Scheduled Sampling for Credit Assignment](https://arxiv.org/pdf/1704.06970.pdf))*: *<u>(more details to be described)</u>*


#### Truncated Backprop

- Concretely, for every $k_1$ time steps, train on the following $k_2$ time steps. When $k_1<k_2$, there's overlap between consecutive time steps; sometimes $k_1=k_2$ is desired.

  ![](https://r2rt.com/static/images/RNN_tf_truncated_backprop.png)

- Initial state may be **zeroed** by a small probability, so as to bias the model towards being easily start from a zero state in test time *(ref: [[Melis, Dyer 2017] On the State of the Art of Evaluation in Neural Language Models](https://arxiv.org/pdf/1707.05589.pdf))*

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

- See also: <https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/> and <https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained>

#### Tied Input and Output Embeddings

- Let $L$ be the input embedding, such that input to the LSTM is $x_t = Ly^*_{t-1}$

- Replace the dense layer $y_t=\mathrm{softmax}(Wh_t+b)$ following the LSTM unit by the **transpose** of the embedding, i.e. $y_t=\mathrm{softmax}\left(L^\top h_t\right)$ *(ref: [[Inan & Khosravi 2016] Tying Word Vectors and Word Classifiers etc.](https://arxiv.org/pdf/1611.01462.pdf))*

- Since input and output are in the same space, it is reasonable to assume they're related by a linear transformation $A$. Tying embeddings results in minimizing w.r.t. vector similarities:

  - Let $u_t=Ly^*_t$, i.e. the embedding of the actual output. By minimizing w.r.t. vector similarities, we would like the probability $y_t$ be related to similarity metrics, concretely $y_t=\tilde{y_t}=\mathrm{softmax}\left(L^\top u_t\right)$, where we use inner product as measurement of similarity.

  - In order to minimize loss, $h_t$ would be adjusted to be closer to the appropriate column of $L$.

  - If we apply KL-divergence as loss, $\tilde{y_t}$ could be used as the estimated true distribution. Other class labels are also utilized during backprop, compared to the case when one-hot encoding is used.

#### Softmax Approximations by Sampling

- Ref to: <http://ruder.io/word-embeddings-softmax/index.html#samplingbasedapproaches>

- Given the correct word $w$, and all candidate words $w_i$. For negative log softmax loss, the formula for loss is:

  $$
  J_w=-\log\frac{\exp(h^\top v_{w})}{\sum_{w_i}\exp(h^\top v_{w_i})}=-h^\top v_w+\log\sum_{w_i}\exp(h^\top v_{w_i})
  $$

  where $v_w$ is the output embedding. Denoting $\mathcal{E}(w)=h^\top v_{w}$, taking gradients w.r.t. parameters would give:

  $$
  \begin{align*}
  \nabla_\theta J_w & = -\nabla_\theta \mathcal{E}(w)+\nabla_\theta \log\sum_{w_i}\exp(\mathcal{E}(w_i)) \\
  & = -\nabla_\theta\mathcal{E}(w)+\sum_{w_i}\frac{\exp(\mathcal{E}(w_i))}{\sum_{w_i'}\exp(\mathcal{E}(w_i'))}\nabla_\theta\mathcal{E}(w_i) \\
  & = -\nabla_\theta\mathcal{E}(w)+\sum_{w_i}P(w_i)\nabla_\theta\mathcal{E}(w_i) \\
  & = -\nabla_\theta\mathcal{E}(w)+\mathbb{E}_{w_i\sim P}[\nabla_\theta\mathcal{E}(w_i)]
  \end{align*}
  $$

  where $P(w_i)$ is the softmax probability of $w_i$.

  Sampling methods reduce computational complexity by approximating the expected term.

##### Importance sampling

  - Expectation can be calculated using Monte Carlo methods: average of samples multiplied by its probability.

  - To avoid computing actual probabilities (which is the same as calculating softmax), sample from another distribution $Q$ similar to the target distribution $P$, for instance, the unigram distribution.

  - Suppose we're to calculate $\mathbb{E}_{x\sim P}[f(x)]$, which in continuous form is equivalent to
    
    $$
    \mathbb{E}_{x\sim P}[f(x)]=\int f(x)p(x)\d x
    $$
    
    where $p(x)$ is the PDF of distribution $P$. We can calculate the integration w.r.t. a different distribution $Q$ with PDF $q(x)$ by evaluating:
    
    $$
    \int f(x)p(x)\d x=\int \frac{f(x)p(x)}{q(x)}q(x)\d x=\mathbb{E}_{x\sim Q}\left[\frac{f(x)p(x)}{q(x)}\right]
    $$
    
    When $Q$ is similar to $P$, doing Monte Carlo integration w.r.t. $Q$ can decrease variance compared to using uniform distribution.

  - To avoid weighting the gradients with $P$, we need to approximate $P$ as well. Denote $P(w)=\frac{\tilde{p}(w)}{Z_p}$, where $Z_p$ is the partition function, and $\tilde{p}(w)=\exp(\mathcal{E}(w))$ is the unnormalized probability of distribution $P$. We can rewrite the expectation as:

    $$
    \begin{align*}
    \mathbb{E}_{w_i\sim P}[\nabla_\theta\mathcal{E}(w_i)] & = \mathbb{E}_{\tilde{w}_i\sim Q}\left[\frac{P(\tilde{w})}{Q(\tilde{w}_i)}\nabla_\theta\mathcal{E}(w_i)\right] \\
     & \approx \frac{1}{m}\sum_{i=1}^{m}\frac{P(\tilde{w}_i)}{Q(\tilde{w}_i)}\nabla_\theta\mathcal{E}(\tilde{w}_i) \\
     & = \frac{Z_q}{Z_p}\frac{1}{m}\sum_{i=1}^{m}\frac{\tilde{p}(\tilde{w}_i)}{\tilde{q}(\tilde{w}_i)}\nabla_\theta\mathcal{E}(\tilde{w}_i)
    \end{align*}
    $$

    where $\tilde{w}_i$ are $m$ samples from distribution $Q$ used in a Monte Carlo estimator. We can apply the same technique in approximating the partition function:

    $$
    \begin{align*}
    \frac{Z_p}{Z_q} & =\frac{1}{Z_q}\sum_{w}\tilde{p}(w) \\
     & = \sum_w \frac{Q(w)}{\tilde{q}(w)}\tilde{p}(w) \\
     & = \mathbb{E}_{w\sim Q}\left[\frac{\tilde{p}(w)}{\tilde{q}(w)}\right] \approx \frac{1}{m}\sum_{i=1}^{m} \frac{\tilde{p}(\tilde{w}_i)}{\tilde{q}(\tilde{w}_i)}
    \end{align*}
    $$

  - Combining the above formulae gives us an unbiased estimator of the expectation:

    $$
    \begin{align*}
    \mathbb{E}_{w_i\sim P}[\nabla_\theta\mathcal{E}(w_i)] & \approx \sum_{i=1}^{m}\frac{\tilde{p}(\tilde{w}_i)/Q(\tilde{w}_i)}{\sum_k \tilde{p}(\tilde{w}_k)/Q(\tilde{w}_k)} \nabla_\theta\mathcal{E}(\tilde{w}_i) \\
     & = \sum_{i=1}^{m}\frac{\exp(\mathcal{E}(\tilde{w}_i))/Q(\tilde{w}_i)}{\sum_k \exp(\mathcal{E}(\tilde{w}_k))/Q(\tilde{w}_k)} \nabla_\theta\mathcal{E}(\tilde{w}_i) \\
     & = \nabla\log\sum_{i=1}^{m}\frac{\exp(\mathcal{E}(\tilde{w}_i))}{Q(\tilde{w}_i)}
    \end{align*}
    $$

    which gives us our actual objective:

    $$
    J_w \approx -\mathcal{E}(w) + \log\sum_{i=1}^{m}\frac{\exp(\mathcal{E}(\tilde{w}_i))}{Q(\tilde{w}_i)}=-\mathcal{E}(w) + \log\sum_{i=1}^{m}\exp(\mathcal{E}(\tilde{w}_i)-\log Q(\tilde{w}_i))
    $$

    The latter form is numerically more stable, and the log-sum-exp trick could be applied.

  - Note that the new objective is also an approximation of the original one. We can see the denominator of softmax as an expectation w.r.t. a uniform distribution, and here we're approximating it with distribution $Q$. But anyway, this is not very accurate, for evaluation, a full softmax is still required.

  - Also refer to [**Implementation Details**](#implementation-details). See also *Pattern Recognition and Machine Learning* Ch. 11.1.4.

##### Noise contrastive estimation (NCE)

  - Ref to: <https://datascience.stackexchange.com/questions/13216/intuitive-explanation-of-noise-contrastive-estimation-nce-loss> and <https://arxiv.org/pdf/1410.8251.pdf>

  - Language modeling can be seen as a multinomial classification problem (predicting the label of the next word). We can convert this into a binary classification problem.

  - Train the LM as is but w/o the final output layer. Jointly train an extra binary classifier to distinguish noise (randomly chosen words) against correct words $w$ given the context $c$.

  - For each word, sample $k$ noises $\tilde{w}_{k}$ from noise distribution $Q$. Minimize per-category cross-entropy loss using logistic regression, giving the loss function, and substituting expectation with Monte Carlo sampling:

    $$
    \begin{align*}
    J_w & \stackrel{\phantom{M.C.}}{=} - \log P(y=1\mid w,c) - k\cdot\mathbb{E}_{\tilde{w}_{j}\sim Q}[\log P(y=0\mid \tilde{w}_{j},c)] \\
    & \stackrel{M.C.}{=} - \log P(y=1\mid w,c) - \sum_{j=1}^{k}\log P(y=0\mid \tilde{w}_{j},c)
    \end{align*}
    $$

    the reason why we used expectation for the noise entropy but not for the positive entropy, is because when we sum over all training data, the positive part would be equal to the entropy calculated for the whole dataset (i.e. the entire distribution).

  - So samples come from a mixture of two distributions: the actual empirical distribution $\tilde{P}$ from data (the distribution we're trying to model), and the noise distribution $Q$. We replace the empirical distribution with the learned distribution $P_\theta$ of our model, which gives:

    $$
    \begin{align*}
    P(w\mid c) & = P(y=0,w\mid c)+P(y=1,w\mid c) \\
     & =\frac{k}{k+1}Q(w)+\frac{1}{k+1}P_\theta(w\mid c) \\
     P(y=1\mid w,c) & = \frac{P(y=1,w\mid c)}{P(w\mid c)}=\frac{P_\theta(w\mid c)}{P_\theta(w\mid c)+k\cdot Q(w)} \\
     P(y=0\mid w,c) & = 1-P(y=1\mid w,c)=\frac{k\cdot Q(w)}{P_\theta(w\mid c)+k\cdot Q(w)}
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

  - **Note**: Performance is poor?

##### Negative sampling

  - An approximation to NCE, by setting the most expensive term $k\cdot Q(w)\equiv1$, giving:

    $$
    P(y=1\mid w,c)=\frac{\exp(h^\top v_w)}{\exp(h^\top v_w)+1}=\frac{1}{1+\exp(-h^\top v_w)}=\sigma(h^\top v_w)
    $$

    where $\sigma$ is the sigmoid function.

  - Equivalent to NCE only when $k=\lvert V\rvert$ and $Q$ is a uniform distribution.

  - **Note**: Inappropriate for language modeling, because probabilistic information is lost. Good for representation learning, as in word2vec.


#### Locality Sensitive Hashing

- A set of hash functions for approximated nearest neighbor search

- **Hyperplane LSH for cosine similarities**: Draw random vectors from normal distribution. For each stored point, check which side of the plane it is at (the sign of their dot product), and encode such information as a 01-string. Such string is used as the hash signature.

- Results are not good.
  - For 10k 128-dim points, best point found by LSH ranked ~23 among actual NNs.
  - This makes it unsuitable for softmax approximations

#### Xavier Initializer & He Initializer

- **Xavier initializer** was proposed by Xavier Glorot, thus also called Glorot initializer *(ref: [[Glorot & Bengio 2010] Understanding the difficulty of training ...](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf))*

- When applying a linear transform $W$ to vector $\mathbf{x}$, we have $\mathbf{y}=W\mathbf{x}=\sum_{i=1}^{n}W_i\mathbf{x}_i$, where $n$ is the dimensionality (or the number of input neurons to the FC layer)

- Assume the input vector has zero mean, and all elements and parameters are IID, we can calculate the variance of $\mathbf{y}$ as follows:
  
  $$
  \begin{align*}
  \mathrm{Var}(\mathbf{y}) & = \mathrm{Var}\left(\sum_{i=1}^{n}W_i\mathbf{x}_i\right)=\sum_{i=1}^{n}\mathrm{Var}(W_i\mathbf{x}_i) \\
   & = \sum_{i=1}^{n}\left(\mathbb{E}[\mathbf{x}_i]^2\mathrm{Var}(W_i) + \mathbb{E}[W_i]^2\mathrm{Var}(\mathbf{x}_i)+\mathrm{Var}(W_i)\mathrm{Var}(\mathbf{x}_i)\right) \\
   & = n\mathrm{Var}(W_i)\mathrm{Var}(\mathbf{x}_i)
  \end{align*}
  $$

- This means the variance is scaled by $n\mathrm{Var}(W_i)$ after the transform. In order to preserve variance, Xavier initializer aims to set the variance of the weights to $\mathrm{Var}(W_i)=\frac{1}{n}=\frac{1}{n_\mathrm{in}}$. If we consider backwards pass, we would find that we need $\mathrm{Var}(W_i)=\frac{1}{n_\mathrm{out}}$, so as a compromise, variance is set to:
  
  $$
  \mathrm{Var}(W_i)=\frac{2}{n_\mathrm{in}+n_\mathrm{out}}
  $$
  
  where $n_\mathrm{in}$ and $n_\mathrm{out}$ corresponds to the dimensions $n$ and $m$ of the transform matrix.

- To obtain such variance, consider a uniform distribution $U[-x,x]$ whose variance is $\mathrm{Var}(U)=\frac{x^2}{3}$. Solving the equation gives us
  
  $$
  W\sim U\Bigg[-\frac{1}{f'(0)}\sqrt{\frac{6}{n+m}},\frac{1}{f'(0)}\sqrt{\frac{6}{n+m}}\Bigg]
  $$
  
  where $f$ is the nonlinearity after the transform.

- **He initializer** was proposed by Kaiming He et al. It simply multiplies the Xavier initializer variance by 2. This is useful for ReLU nonlinearities, whose derivative is undefined at 0. This also makes sense in ReLU's derivative is 0 half the time and 1 for the other half. *(ref: [[He 2015] Delving Deep into Rectifiers...](https://arxiv.org/pdf/1502.01852))*

- Ref to: <http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization>

#### Tuning on the Development Set

- **Early-stopping**: When results does not get better on the dev set, simply stop training. Usually there's a threshold (or **patience**) as to how many epochs with worse results are tolerated.

- **Rollback**: When results does not get better, simply load the best previous model and decay the learning rate. Best models as usually saved to disk.

- **Rollback Optimizer**: Regarding the optimizer (we're only concerned about the optimizer statistics, e.g. parameter momentum, but not the hyper-params e.g. learning rate), 3 strategies are possible:

  1. Load the optimizer at the time of the best model snapshot. This requires saving the optimizer state as well.
  2. Reset optimizer statistics. For common optimizers this mean zeroing momentum and moments.
  3. Use the current optimizer as-is. The current optimizer has statistics for the worse dev performance part of training.

  Usually the effectiveness of the 3 methods are in order of their numbering. However method 2 has the benefit of being able to escape out of local minima as the initial step would be a large step.


## Theory & Proofs

#### On Gradient Vanishing/Exploding of RNNs

- *(ref: [[Zilly 2016] Recurrent Highway Networks](https://arxiv.org/pdf/1607.03474), chapter 2)*

A vanilla RNN can be described as

$$
y^{(t)}=f\left(Wx^{(t)}+Ry^{(t-1)}+b\right)
$$

For simplicity, suppose the loss is defined on the last state only, i.e. $\mathcal{L}=g\left(y^{(T)}\right)$. The gradient w.r.t. parameters would be

$$
\frac{\d\mathcal{L}}{\d\theta}=\frac{\d\mathcal{L}}{\d y^{(T)}}\frac{\d y^{(T)}}{\d \theta}=\frac{\d\mathcal{L}}{\d y^{(T)}}\sum_{t_1=1}^{T}\frac{\d y^{(T)}}{\d y^{(t_1)}}\left(\frac{\d y^{(t_1)}}{\d W}+\frac{\d y^{(t_1)}}{\d b}\right)
$$

In the formula above, the gradient is expanded using the chain rule, and then expanded along the time axis. We further expand the Jacobian $\frac{\d y^{(T)}}{\d y^{(t_1)}}$:

$$
\frac{\d y^{(T)}}{\d y^{(t_1)}}=\prod_{t_1<t\leq T}\frac{\d y^{(t)}}{\d y^{(t-1)}}=\prod_{t_1<t\leq T}R\cdot\mathrm{diag}\left[f'\left(Ry^{(t-1)}\right)\right]
$$

Denoting $A=\frac{\d y^{(t)}}{\d y^{(t-1)}}$ as the temporal Jacobian, the upper bound for its norm would be

$$
\Vert A\Vert\leq \Vert R\Vert \left\Vert f'\left(Ry^{(t-1)}\right)\right\Vert\leq \sigma_\max\cdot\gamma
$$

where $\sigma_\max$ is the principal singular value of $A$, and $\gamma$ is the upper bound on $f'$.

As the temporal Jacobian is multiplied together, which is approximately equivalent to $A$ raised to the $T$-th power. So the conditions for vanishing/exploding gradients are:

- **Vanishing gradients**: $\gamma\sigma_\max<1$

- **Exploding gradients**: $\rho(A)=\sigma_\max>1$

**Comparing to MLPs**: The reason deep CNNs/MLPs do not suffer less from gradient vanishing/exploding as RNNs do, is because MLPs use different matrices at different layers, while RNNs use the same matrix in every time step.

#### On the Effectiveness of LSTMs

- *(ref: [[Jozefowicz 2015] An Empirical Exploration of RNN structures](http://proceedings.mlr.press/v37/jozefowicz15.pdf), chapter 2)*

In their simplest forms, the RNN calculates the new state $h^{(t)}$ by $h^{(t)}=f(Wh^{(t-1)})$, while the LSTM (without forget gates) calculates the new state by $c^{(t)}=c^{(t-1)}+i_\mathrm{g}^{(t)}f(Wc^{(t-1)})$, $h^{(t)}=o_\mathrm{g}^{(t)}c^{(t)}$.

The temporal Jacobian here would be

$$
\frac{\d c^{(t)}}{\d c^{(t-1)}}=1
$$

Or to put simply, to obtain the state at time step $t$, RNNs would apply $t$ times the transformation $f$, while LSTMs calculate the increment at each time step and sums them up.




## Implementation Details

#### About DyNet

- Transpose (`dy.tranpose`) requires making a copy of the matrix, so does `dy.concatenate_cols` and similar functions.

- `lstm.disable_dropout` does not work, use `lstm.set_dropouts(0, 0)` instead.

- Load parameters of LSTM initial state (as in truncated backprop) by:

  ```python
  s = [vec.npvalue().reshape(-1, batch_size) for vec in state.s()]
  # dy.renew_cg()
  state = self.lstm.initial_state().set_s([dy.inputTensor(vec, batched=True) for vec in s])
  ```

- Use `dy.affine_transform([b, W, x])` for linear layer with biases, this is more efficient.

- `dy.log_softmax` is more efficient than `dy.log(dy.softmax(x))`, and prevents numerical problems. Similarly, `dy.pickneglogsoftmax` is better than `dy.log_softmax` then `dy.pick`.

- Note the difference between `dy.pick`, `dy.pick_batch` and `dy.pick_batch_elem`.

#### Log-probability Domain

- Addition in log domain is done by the log-sum-exp operation $\ln\sum\exp(x_i)$.

- DyNet has `dy.logsumexp`, and so does Numpy.

- See computation tricks at <https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/>

#### Importance Sampling

- There's a chance that the ground-truth $w$ is not included in the approximation of the denominator. This could lead to a negative which is bad for optimization.

- For a biased solution, we can simply include all targets in the samples. Although effective, this leads to a biased estimator, and deprives the loss of its probabilistic information (it cannot be used to evaluate perplexity).

- For an unbiased solution, we can modify the proposal distribution to bias towards to targets. What differs from the above solution is that we still *sample* from the distribution, rather than forcibly modifying the samples.

- For efficient calculation, we can use the same samples are shared across the mini-batch and time steps. *(ref: [[Jozefowicz et al. 2016] Exploring the Limits of Language Modeling][https://arxiv.org/pdf/1602.02410])*




## Models & Structures

#### [Latent predictor networks](https://arxiv.org/pdf/1603.06744.pdf) by W. Ling et al

- When using a normal RNN model, after inference is made, we can backtrack from the final state through to the initial state, resulting in a path

- This is due to that RNNs generated one token at a time, and at each time step samples the next token according to calculated probabilities

- If RNNs can generate multiple tokens in one time step, there may be <u>multiple paths from the initial state to the target state, corresponding to segmentations of the sequence</u>. Path counts can be exponential to its length, and the union of the paths is a directed acyclic graph

- Paper proposes a method to perform <u>joint training on several predictors of different granularity</u>. The method introduced latent variables for deciding which predictor to use, thus giving it the name

- To calculate gradients for a time step, <u>summed products of probabilities on the DAG</u> are required, which can be calculated using a dynamic programming algorithm

- Attention over all different fields in a structured input is used

- Prediction is done using beam search

- The authors utilized this technique in code generation tasks for card games, where a character-level LSTM predictor is jointly trained with pointer networks for copying text directly from card descriptions

#### <u>Neural lattice language models</u> by Jacob (Graham's grad. student)

- Main idea is similar: enabling LSTM models to <u>generate multiple tokens in one time step</u>

- Exact probability is hard to calculate as LSTMs keep track of whole context seen from the initial state, so each path would have a different state

- Paper evaluated different approaches of probability estimations and different representations of multiple-tokens in one time step *<u>(more details to be described)</u>*

  - Ancestral sampling from "latent sequence decompositions": just treat multiword tokens as regular tokens

  - TreeLSTM-style summation: summing predecessors' hidden states. Cons: losses probabilistic info

  - Weighted expectation: weight summations using prob. dist. learned in ancestral sampling

- **Difference with "<u>latent predictor networks</u>"**:

  - Latent predictor networks combine multiple predictor models, while this is one unified model

  - The reason why probabilities are easy to calculate in said paper is due to the fact that, although predictors of different granularity are used, <u>all predicted tokens are in the same space, and multiple tokens are fed into the character-level network one-by-one</u>. Hidden states of the char-level network is used in the pointer network in turn. So only O(length) states are required in total

#### [Pointer Networks](https://arxiv.org/pdf/1506.03134) by O. Vinyals et al

- Output is the set of tokens from input, instead of fixed vocabulary

- Basically a seq2seq model with attention, but use attention weights directly as probability from predicting each input token

- Can be trained to select ordered subsets from input, even accomplish difficult tasks as convex hulls, Delauney triangulation and TSP

- See also: <http://fastml.com/introduction-to-pointer-networks/>

#### [TreeLSTMs](https://arxiv.org/pdf/1503.00075.pdf) by K. S. Tai, R. Socher, and Christopher D. Manning

- A natural generalization of LSTM to tree structures

- Sum children hidden states as $\tilde{h}$, and replace this as $h$ in formulas for normal LSTMs

- Forget gate is different for each child: use only the hidden state of child to calculate forget gate parameters

- Cell state of parent is as usual, summing over cell states of each child with respective forget gates

- Ordered children version exists: use different parameters for each child (depending on its index). Such model has a limit on the maximum branch factor

- **Benefits**: Can make use of sentence structures generated by parsers; better at preserving state, i.e. can cope better with long distance dependencies (since path lengths are shorter on trees)

#### [Adapted Softmax](https://arxiv.org/pdf/1609.04309.pdf) by Grave et al

- Efficient for large vocabularies, and optimized according to empirical analysis of matrix multiplication speed on GPUs:
  - For matrices with dimensions $k$, the empirical formula for matrix multiplication time cost is $g(k)=c_\mathrm{m}+\max(0,\lambda(k-k_0))$, where typically $c_\mathrm{m}=0.40\ \mathrm{ms}$, and $k_0=50$.

- Proposed structure is a two level hierarchical softmax:
  - First level contains all common words (~20% words covering ~80% corpus), and representations of clusters
  - Remaining words are grouped into clusters according to frequency. Words with lower frequency fall into larger clusters.

#### [Highway Networks](https://arxiv.org/pdf/1505.00387) by Srivastava et al

- Simply put, **Highway Networks** add LSTM-style input (a.k.a. transfer) and forget (a.k.a. carry) gates to normal NN layers. Proposed structures uses tied gates (i.e. $f_\mathrm{g}=1-i_\mathrm{g}$). Such structure can be applied to very deep NNs to help training.
- In practice, transfer gates are initialized with a negative bias to bias the network towards carry behavior. The intuition is the same as initialize forget gates biases to 1 or 2 to enable gradient flow at early stages and preserve long term memories *(ref: [[Gers 1999] Learning to Forget...](https://pdfs.semanticscholar.org/1154/0131eae85b2e11d53df7f1360eeb6476e7f4.pdf))*.




## Topic: Language Models

#### [Character-Aware Neural LM](https://arxiv.org/pdf/1508.06615) by Kim et al

- Each word is fed into a character-level CNN:
  - Append SOW and EOW tokens to each word (and zero-pad for batching)
  - Apply 1-d convolution to low-dimension char embeddings
  - Max-pool over all features for each feature map, and concatenate

- Output is fed through a highway layer, i.e. feeding a part of the vector directly through to the output, and transforming the rest.

- Output is then fed into an LSTM-LM, making word-level predictions. Two-layer hierarchical softmax is used for large vocabularies.

- **Pros & Cons**:
  - Fewer params than vanilla LSTM-LMs (due to the absence of word vectors).
  - Can deal with OOV inputs, but not outputs; good for morphologically-rich languages.
  - Computationally more expensive.



## Topic: Embeddings

#### <u>word2vec</u> by Mikolov et al

- Reference papers: [CBOW & Skip-gram models](https://arxiv.org/pdf/1301.3781.pdf), [Negative sampling](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf); see also: [CS224n Lecture Notes 1](http://web.stanford.edu/class/cs224n/lecture_notes/cs224n-2017-notes1.pdf)

- **CBOW** predicts the center word given surrounding (context) words. **Skip-gram** predicts surrounding words given the center word.
  Both methods learn two representations $\mathcal{U}$ (output) and $\mathcal{V}$ (input) for each word, generating output probabilities by $\hat{y}=\mathrm{softmax}(\mathcal{UV}x)$, where $x$ is a one-hot vector (for skip-gram, or average of one-hot vectors for CBOW).
  - The "two representation" part, in its essence, is a rank constraint on the matrix. Or in other words, a down projection.

- For **negative sampling**, refer to [previous section](#negative-sampling).

- Intuition for vector additive compositionality (e.g. skip-gram):
  - Due to the linearity in the training objective, summing two vectors would result in summing log probabilities in the output layer.
  - This is equivalent to the product of context word distributions, so words appearing near both words have higher probabilities.

- About the impact of **window size** on word representations:
  - Representations learnt with smaller windows is more aware of **syntactic** relations.
  - With larger windows, **semantic** relations are well captured.

- With **subword information** *(ref: [[Bojanwoski 2016] Enriching Word Vectors with Subword Information](https://arxiv.org/pdf/1607.04606))*
  - Replace word embeddings in skip-gram with the sum of n-gram embeddings.
  - Boundary symbols are added to begin and end of words

- **[fastText](https://arxiv.org/pdf/1607.01759)** classifier:
  - Bag of words + Bag of bigrams
  - Feeds average of word/n-gram embeddings through a FC layer
  - Uses very low dimensions (10 for sentiment, 50-200 for tags), very fast

#### [context2vec](http://aclweb.org/anthology/K16-1006) by Melamud et al

- Run a Bi-LSTM on the sentence, and use the prefix and suffix hidden state vectors from both LSTMs as the context representation for a given word.

- **Compared to word2vec**: The two models basically does the same thing. As context2vec uses LSTMs to generate context reps, it is capable of handling larger contexts and deal with long distance relationships.

#### [Paragraph Vector](https://arxiv.org/pdf/1405.4053) by Le & Mikolov

- Assign embeddings to each word and each paragraph. For each window, concat the paragraph vector and the first few word vectors and feed through an FC layer predict the last word. (Similar to CBOW)

- To generate representation for a new paragraph, train the vector while freezing other parameters.

- Another method (similar to skip-gram): sample a random window from the paragraph, and predict words in the window given the paragraph vector.

- Vectors learnt by the two methods are concatenated for use in downstream tasks.

#### [C2W](http://www.cs.cmu.edu/~lingwang/papers/emnlp2015.pdf) (character to word) by W. Ling et al

- Generate word representations by running a char-level Bi-LSTM.

- Author does not generate embeddings directly in an unsupervised fashion; this model is used only in downstream tasks (POS-tagging, LM).

#### [Tensor Indexing Model](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9597/9526) by Y. Zhao & Zhiyuan Liu

- Find common two-word phrases from corpus and replace them by a single phrase token.

- Train using skip-gram objective, obtaining embeddings in the same space for the whole phrase and words in the phrase.

- Train a composition model approximating the phrase embedding $\mathbf{z}$ given word embeddings $\mathbf{x}$ and $\mathbf{y}$:
  $$
  \mathbf{z}_i=f(\mathbf{x},\mathbf{y})=\mathbf{x}^\top W_i\mathbf{y}+(M\mathbf{x})_i+(N\mathbf{y})_i
  $$
  and apply rank constraints on matrices $W_i$, giving $W_i\approx U_i^\top V_i+I$.

- Loss function is MSE of $\mathbf{z}$ and $f(\mathbf{x},\mathbf{y})$, plus a regularization factor.

#### [Structured Word2Vec](http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf) by W. Ling et al

- Simple modifications to original word2vec model, taking **word order** into consideration.

- **Structured Skip-gram**: Use different matrices for each context position.

- **Continuous Window**: Concat context vectors instead of summing them.

- Such models are better at capturing syntactic information.

#### [sembei](http://aclweb.org/anthology/D17-1081) (segmentation-free word embeddings) by T. Oshikiri

- Based on skip-gram, trains n-gram embeddings.

- Only consider cases when the center and context n-grams are adjacent, use separate matrices for left and right contexts (as in [Structured Word2Vec](#structured-word2vec-by-w-ling-et-al)).

  - Theoretically capable of dealing with arbitrary sized windows, a sample is regarded as positive as long as there exists such a segmentation of the corpus into the chosen n-grams. But this is computationally expensive.
  
- **Limitations**: N-grams are the model's vocabularies, i.e. it provides no means of composition, thus cannot deal with OOV words.

## Topic: Dependency Parsing

#### Task Description

- Generate a tree structure over a sentence, describing dependency of words.
- A directed tree, with a label for each edge, denoting the type of dependency.
- There can be multiple roots for a single sentence, especially when the sentence is conjoined by conjunctions (e.g. "and").

#### Methods

- Methods can be basically categorized into one of the two:

- **Transition-based methods**: Shift-reduce parsers that acts like a pushdown automata, at each time step the model can choose to either SHIFT (push word to stack), or REDUCE (pop the top two words from stack, assign one of the as the parent of the other, and push the parent back into the stack). More complicated models support other actions. These methods often make local decisions, and use greedy decoding.

- **Arc-factored graph-based methods**: Assign likelihood for each ordered pair of nodes, and for each label type. Run Chu-Liu-Edmonds' algorithm for maximum spanning arborescence. These methods usually make global decisions.

#### [Deep Biaffine Attention](https://arxiv.org/pdf/1611.01734) by T. Dozat & Manning

- Refer to code implementations: <https://github.com/chantera/teras/blob/master/teras/framework/pytorch/model.py> and <https://github.com/chantera/biaffineparser/blob/master/pytorch_model.py> (two parts of one code)

- An arc-factored graph-based method. Similar to self-attention.

- Runs a Bi-LSTM over the sentence, then passes each state through 4 separate MLPs, to generate embeddings for the word, when used as both dependant and head, and when used in predicting arc head and edge label.

  - Recurrent state: $\mathbf{r}_i$

  - Embeddings: $H^{({\text{arc-dep}})} = [\cdots \mathbf{h}_i^{(\text{arc-dep})} \cdots]$, $\mathbf{h}_i^{\text{arc-dep}} = \mathrm{MLP}^{(\text{arc-dep})}(\mathbf{r}_i)$.

    - and similarly for "arc-head", "label-dep", and "label-head".
    - This serves as dimensionality reduction.

  - Arc scores for word $i$ (i.e. the likelihood of each word being the dependency head of $i$), vector length = sentence length:
    
    $$
    \begin{align*}
    \mathbf{s}_i^{(\text{arc})} & = {H^{(\text{arc-head})}}^\top \left( U^{(\text{arc-1})}\mathbf{h}_i^{(\text{arc-dep})} + \mathbf{u}^{(\text{arc-2})} \right) \\
     & = {H^{(\text{arc-head})}}^\top U_b^{(\text{arc})}
            \left[\begin{matrix} \mathbf{h}_i^{(\text{arc-dep})} \\ 1 \end{matrix}\right]
    \end{align*}
    $$

  - Score of label type $k$, for word $i$, given its <u>true head</u> $y_i$, vector length = number of label types:
 
    $$
    \begin{align*}
    \mathbf{s}_i^{(\text{label}_k)} & = {\mathbf{h}_{y_i}^{(\text{label-head})}}^\top U_k^{(\text{label-1})} \mathbf{h}_i^{(\text{label-dep})} + {\mathbf{u}^{(\text{label-2,1})}}^\top\mathbf{h}_i^{(\text{label-dep})} + {\mathbf{u}^{(\text{label-2,2})}}^\top\mathbf{h}_{y_i}^{(\text{label-head})} + b_k \\
     & = \left[\begin{matrix} \mathbf{h}_{y_i}^{(\text{label-head})} \\ 1 \end{matrix}\right]^\top
           U_b^{(\text{label})}
           \left[\begin{matrix} \mathbf{h}_i^{(\text{label-dep})} \\ 1 \end{matrix}\right]
           - 1 + b_k
    \end{align*}
    $$

  - Scores are equivalent to affine transformations on two vectors, hence the name.

  - Loss = neg log softmax of correct head for each node, plus neg log softmax of correct type for correct head for each node.

- Needs to run Chu-Liu's algorithm during prediction. Select argmax type for each predicted edge.

- When the formulae are expanded, they take the same form as the [Tensor Indexing Model](#tensor-indexing-model-by-y-zhao--zhiyuan-liu)


## Topic: Variational Auto-encoder

#### Formulation of VAE

- Ref to: [Tutorial on Variational Autoencoders](https://arxiv.org/pdf/1606.05908)

##### The first formula

- To build a generative model, we need to approximate the distribution $P(X)$ where $X$ is our data (things to generate).

- An intuitive method is to first extract features $\mathbf{z}$ from $X$'s, and use a model parameterized by $\theta$ to recover $X$ given $\mathbf{z}$.

- The features $\mathbf{z}$ are called **latent variables** (means "hidden") because they're not *observed* but *inferred*.

- Following the law of total probability, we have

  $$
  P(X) = \int P_\lambda(\mathbf{z}) P_\theta(X\mid\mathbf{z})\d\mathbf{z} = \mathbb{E}_{\mathbf{z}\sim P_\lambda}[P_\theta(X\mid\mathbf{z})]
  $$

  where $P_\lambda(\mathbf{z})$ is our **prior** knowledge of the space of latent variables, and $P_\theta(X\mid\mathbf{z})$ if the likelihood approximated by our model, parameterized by $\theta$. In modern context, $\theta$ can be seen as the **decoder**.

- Naturally, our objective would be to maximize the expectation of the marginal log-probability over the data distribution, $\mathbb{E}_{X\sim D}[\log P(X)]$. This is a form of maximum likelihood estimation (MLE).

##### Simplify calculations: Introducing posterior

- But such integration is intractable for three reasons:

  - The space of $\mathbf{z}$ is large.
  - Monte Carlo sampling would not be effective because $P_\theta(X\mid\mathbf{z})$ is likely to be zero for most $\mathbf{z}$'s.
  - We want to run optimization in mini-batches. Consider maximizing the objective w.r.t. a single data example $X$, we're effectively increasing the probability of $X$ given any latent $\mathbf{z}$. This is clearly counter-intuitive.

- From the second reason, it is natural to consider using **importance sampling** to speed up sampling procedure. Thus we introduce a distribution $Q_\phi(\mathbf{z}\mid X)$, which gives high probability to latent $\mathbf{z}$'s that would in turn give high probability to the example $X$. This is the **posterior** distribution matching its prior. In modern context, $\phi$ can be seen as the **encoder**.

- With the posterior in mind, we can rewrite the objective

  $$
  P(X) = \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}\left[ \frac{P_\lambda(\mathbf{z})P_\theta(X\mid\mathbf{z})}{Q_\phi(\mathbf{z}\mid X)} \right]
  $$

##### Simplify calculations: Log-probability domain & KL-divergence

- To match our objective, we try transform everything into the log-probability domain. But the logarithm function cannot be moved inside the expectation.

- However, **Jensen's inequality** states that for any convex function $f$ and random variable $X$, we have $f(\mathbb{E}[X]) \leq \mathbb{E}[f(x)]$. For concave functions like $\log$, the opposite conclusion holds. Thus we derive a lower bound on $\log P(X)$:

  $$
  \begin{align*}
  \log P(X) & \geq \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}\left[ \log\frac{P_\lambda(\mathbf{z})P_\theta(X\mid\mathbf{z})}{Q_\phi(\mathbf{z}\mid X)} \right] \\
   & \geq \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}\left[ \log P_\lambda(\mathbf{z}) + \log P_\theta(X\mid\mathbf{z}) - \log Q_\phi(\mathbf{z}\mid X) \right]
  \end{align*}
  $$

- Observe that the above formulation can be rewritten as

  $$
  \log P(X) \geq \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}[\log P_\theta(X\mid \mathbf{z})] - \mathrm{KL}(Q_\phi(\mathbf{z}\mid X)\Vert P_\lambda(\mathbf{z}))
  $$

  where the second term is the KL-divergence given by $\mathrm{KL}(q\Vert p) = H(p,q)-H(q)$, the difference of cross-entropy and entropy.

  The RHS is also known as **Evidence Lower Bound** (ELBO) $\mathcal{L}(X;\phi,\theta,\lambda)$.

- The lower bound can be interpreted as:

  - Maximizing the likelihood w.r.t. the posterior of latent $\mathbf{z}$ given the trained example $X$, and
  - Regularizing the posterior distribution by pulling it close to the prior.

##### How do we compute the lower bound?

- We should first define the form for our prior and posterior. A common choice is to use a Gaussian distribution (or a mixture of Gaussians). There are two reasons for this:

  - For the prior, an univariate Gaussian distribution defined on $\mathbb{R}$ is able to represent any distribution by composing the inverse CDF of the desired distribution with the CDF of a Gaussian. This also holds true for multiple dimensions. So we can trust our decoder to learn this mapping, which shouldn't be too difficult for neural network models.
  - For the posterior, using a Gaussian distribution means that we only have to specify the mean $\mu$ and variance $\sigma^2$. Also, such settings gives an analytical solution to the KL-divergence term.

- For the first term, given the assumption that $Q_\phi$ is able to produce a nice estimate of the posterior, we can safely use Monte Carlo sampling, i.e. sample $\mathbf{z}$ from distribution $Q_\phi$, and optimize for $\log P_\theta(X\mid\mathbf{z})$.

  - However, in terms of SGD, this is not acceptable, because "sampling" is indifferentiable.
    - To see why this is true, note that for the term we backprop w.r.t. the negative log likelihood, which is dependent only on $\mathbf{z}$.
    - However, $\mathbf{z}$ is sampled from $Q_\phi$, but it is introduced into $P_\theta$ as "input", which has no gradient. So the gradient is cannot be backpropped through the sampling procedure.
  - So a re-parameterization trick must be applied: since $Q_\phi$ is Gaussian $\mathcal{N}(\mu,\sigma^2)$, sampling $z\sim \mathcal{N}(\mu,\sigma^2)$ is equivalent to sampling $\epsilon\sim\mathcal{N}(0,1)$ and compute $z=\mu + \sigma\cdot\epsilon$. Thus gradient is able to flow through the encoder. *(ref: [[Kingma & Welling 2013] Autoencoding Variational Bayes](https://arxiv.org/pdf/1312.6114))*

- For the second term, given Gaussians $P(\mathbf{z})=\mathcal{N}(0,I)$ and $Q_\phi(\mathbf{z}\mid X)=\mathcal{N}(\pmb{\mu}(X),\pmb{\Sigma}(X)=\mathrm{diag}(\pmb{\sigma}^2(X)))$, denoting $n$ as the dimensionality of $\mathbf{z}$, KL-divergence has the following analytical form
  $$
  \mathrm{KL}(Q_\phi(\mathbf{z}\mid X)\Vert P(\mathbf{z})) = \frac{1}{2}\sum_{i=1}^{n}\left(1+\log(\sigma_i^2) - \mu_i^2 - \sigma_i^2\right)
  $$

##### When does equality hold true?

- To investigate the problem, subtract the RHS from LHS. But first, we make the assumption that our parameterized model is able to model the ground truth likelihood. Thus in the following deduction, we omit the subscripts on $P$:

  $$
  \begin{align*}
   & \phantom{=\;\;\!} \log P(X) - \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}\left[ \log P(\mathbf{z}) + \log P(X\mid\mathbf{z}) - \log Q_\phi(\mathbf{z}\mid X) \right] \\
   & = \log P(X) - \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}\left[ \log P(\mathbf{z}) + \left(\log P(\mathbf{z}\mid X) + \log P(X) - \log P(\mathbf{z})\right) - \log Q_\phi(\mathbf{z}\mid X) \right] \\
   & = \mathbb{E}_{\mathbf{z}\sim Q_\phi(\cdot\mid X)}[\log Q_\phi(\mathbf{z}\mid X) - \log P(\mathbf{z}\mid X)] = \mathrm{KL}(Q_\phi(\mathbf{z}\mid X)\Vert P(\mathbf{z}\mid X))
  \end{align*}
  $$

  where $P(\mathbf{z}\mid X)$ is the true posterior. So the closer $Q_\phi$ matches the true posterior distribution, the tighter our lower bound is.

- This also shows another reason why we choose to optimize the lower bound instead: this KL-divergence term is intractable, because we have no idea which $\mathbf{z}$'s give high probability to $X$.

##### FAQ

With the knowledge in mind, we can look back at some problems that was glossed over:

- **Why is a simple Gaussian sufficient for prior?** Apart from the "NN can learn any CDF" reason, we use Gaussian also because it is a commonly-used distribution with a non-zero probability for every point in $\mathbb{R}$. And the fact that not all distributions can be re-parameterized.
- **Why do we constrain the KL-divergence of posterior $Q_\phi$ and prior $P_\lambda$?** This does not make sense in that if this term is minimized, then the KL-divergence term on the other side of the equation, namely $\mathrm{KL}(Q_\phi(\mathbf{z}\mid X)\Vert P(\mathbf{z}\mid X))$ would be large, which gives us a loose lower bound. This term mainly serves as **regularization**, for we're using NNs for $Q_\phi$, and we should constrain its form.

#### Possible issues

- **Over-regularization**
  - KL-divergence takes a simple form for simplistic priors, and is much easier to learn. Encoder would quickly match the Gaussian prior.
  - **Solutions**:
    - Initially set KL-divergence term to zero, and gradually anneal to a predefined scale. Can be seen as first overfitting and then regularizing.
    - Or, design more complex priors.
- **Ignoring latent code**
  - For sequential decoders, a simple latent code would force the model to rely on the ground truth of previous time steps, and a powerful model may learn decoding without consulting the latent code.
  - **Solutions**:
    - Apply dropout on decoder inputs.
    - Or, constraining the amount of context that the decoder is allow to see.

#### Extensions

##### Conditional VAE

- Autoencode $X$ given $Y$, for instance generate user content given his previous work.
- Simply change $Q_\phi$ and $P_\theta$ to conditional distributions. This means both the encoder and decoder needs to condition on $Y$.

##### Discrete Latent Variables

- Reparameterization trick fails for discrete distributions.

- Marginalize over every possible discrete choice.

- Or, use the [Gumbel-Softmax](#gumbel-max-trick--gumbel-softmax-distribution) technique.

  ​
