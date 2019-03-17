

#Hierarchal Softmax (Bengio) 

https://www.iro.umontreal.ca/~lisa/pointeurs/hierarchical-nnlm-aistats05.pdf 3. 

[https://arxiv.org/pdf/1609.04309.pdf] 2 

https://arxiv.org/pdf/cs/0108006.pdf 1 



Part one: logarithmic speedup in softmax based on word-clustering. 

- model 1: predict class of word given context $P(class(w) | w_1 … w_{i-1}) ​$ 
- model 2: predict word given class and context, $P(w | w_1 … w_{i-1}, class(w_i))​$ . 



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

given a deterministic function $c(.)$ mapping $Y$ (the output labels) to $C$ (clustering partition), decompose problem into:
$$
P(Y=y|X=x) = \\
P(Y=y | C=c(y), X)P(C=c(y) | X = x).
$$
!!! todo: think more deeply about the connection between the math and the actual practice. 




$$
P(C = c(y) | X=x)
$$
What are you normalizing here? 

Essentially we need parameters that tell let us calculate
$$
\text{argmax}_{c(y)} P(C= c(y) | X = x)
$$


My understanding: 

- normalize over cluster partition probabilities
- normalize over target probabilities 




$$
!!!
$$


# Hierarchal Softmax (FAIR)



# Adaptive Softmax

- exploit unbalanced word distribution to reduce dependency on vocabulary size. 
- 



# Adaptive Input Embeddings (FAIR)



