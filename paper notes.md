

#Hierarchal Softmax (Bengio)

https://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf 3. 

[https://arxiv.org/pdf/1609.04309.pdf] 2 

https://arxiv.org/pdf/cs/0108006.pdf 1 



Paper: [Classes For fast maximum entropy training](https://arxiv.org/pdf/cs/0108006.pdf)



**Part one**: logarithmic speedup in softmax based on word-clustering. 

- model 1: predict class of word given context $P(class(w) | w_1 … w_{i-1}) $ 
- model 2: predict word given class and context, $P(w | w_1 … w_{i-1}, class(w_i))$ . 



Language Modeling is essentially a probability density over words. One bottleneck in neural language models is the softmax layer (final layer), which must assign a probability over all words. The softmax, is 


$$
\sigma(w_i) = \frac{\exp(w_i) }{\sum_j{w_j} }
$$


- don't need to compute all word probabilities
- (language model is a probability density over words)

- exploit smoothness of neural networks to make sure sequences of similar words are assigned similar probability. 



- rewrite a probability function based on a partition of the set of words. In other words, partition softmax function into distinct sets. 
- Determine word class hierarchy by the dataset in question. 



Computing the softmax is costly, because in order to calculate the probability densities for the softmax, the denominator scales with $|V|$ linearly, which is pretty bad if $|V|$ is gigantic, such as in NLM. 



So this paper essentially makes the $O(|V|)$ computation into a $O(\log |V|)$ one. 





## Hierarchal decomposition - exponential speedup. 

given a deterministic function $c(.)$ mapping $Y$ (the output labels) to $C​$ (clustering partition), decompose problem into:
$$
P(Y=y|X=x) = \\
P(Y=y | C=c(y), X)P(C=c(y) | X = x).
$$
!!! **todo**: think more deeply about the connection between the math and the actual practice. 




$$
P(C = c(y) | X=x)
$$
What are you normalizing here? 

Essentially, we need parameters that tell let us calculate
$$
\text{argmax}_{c(y)} P(C= c(y) | X = x)
$$


My understanding: 

- you now train two models: one to predict $P(C | w_1, … w_{i-1})​$, the second to predict word given class and context. 
- normalize over cluster partition probabilities
- normalize over target probabilities



??? what if you get the class is wrong?

??? so you first predict the class, and then given the class, you predict the word?

# Hierarchal Softmax (FAIR)



# [Adaptive Softmax (fair)](https://arxiv.org/pdf/1609.04309.pdf)

code: https://github.com/facebookresearch/adaptive-softmax



key idea:

- exploit unbalanced word distribution to reduce dependency on vocabulary size. 

- exploit matrix-matrix operations for fast GPU runtime. 



**An approximate hierarchical model**



### Computation time model of matrix multiplication



# Adaptive Input Embeddings (FAIR)

Hidden states (size $B \times d$) (B: batch size, d: hidden layer size)

word representation: (size $d \times k$) 

???But why would you do that??? 

Let $g(k, B)$ be the computation time for this multiplication. The observation is that $g(k)$ is constant for low values of $k$, until a certain inflection point $k_0 \approx 50$, subsequently becoming affine for values $k > k_0$ 



We can model the runtime of matrix-matrix multiplication 
$$
g(k) = \max (c + \lambda k_0, c + \lambda k) \\
= c_m + \max [0, \lambda(k - k_0)]
$$
The intuition for this is as following: 

It's inefficient to matrix-matrix multiplication when one of the dimensions is small. Thus, hierarchical softmaxes where one of the dimensions is small is very inefficient. In addition, clusters with only rare words have low probability $p$ and small batch size of $B$ ,which leads to inefficient matrix-matrix multiplication. 

**Instead!!!** Use this format: 


$$
g(k, B) = \max (c + \lambda k_0B_0, c + \lambda kB)
$$
?? i don't get why but sure



### Two clusters case



87% of NLP language vocab is covered by 20% of vocab. Simple strategy to reduce comptutation time: 

partition $\mathcal {V}$ into $\mathcal{V_h, V_t}$. Where $h$ refers to the head, $t$ refers to tail. 

Define clusters with unbalance cardinalities $\vert \mathcal{V_h} \vert \ll \vert \mathcal{V_t}\vert $ 

Define unbalanced probabilities $P (\mathcal{V_h} )  \gg \vert P(\mathcal{V_t})$



### Short List



Idea: 

![Screen Shot 2019-03-18 at 9.59.38 PM](/Users/edgar/Desktop/screenshots/Screen Shot 2019-03-18 at 9.59.38 PM.jpg)



in $V_h$, the shortlist is the first 3 blue, which directly have their words as leaves, where the tail end cluster has subclusters. Sizes determined as to minimize computational model on GPU. 



