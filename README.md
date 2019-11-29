# GPMF-GBP-AAAI-20
This MATLAB code is for demonstration experiments, using Movielens 100k dataset with side information, for the paper:

Strahl, J., Peltonen, J., Mamitsuka, H., & Kaski, S. (2020). Scalable Probabilistic Matrix Factorization with Graph-Based Priors. To appear in Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI-20), preprint on arXiv.

Our method, Graph-Based Prior Probabilistic Matrix Factorisation (GPMF), estimates the unobserved ratings for the Movielens data, given the demographic side-information for users and genre side-information for movies.  A kNN graph is created from the feature side-information used as regularisation.  

GPMF initialises the latent features with no side-information using PMF (GRALS with empty graphs).  Next the M-step of the algorithm identifies edges in the graph that are contested (disagree with) the correlations between latent feature vectors. If two latent features have a negative correlation, the edge in the graph is contested, and therefore removed.  Finally the E-step runs the GRALS algorithm with the updated graph with contested edges removed.

ACKNOWLEDGEMENTS:

Function for generating samples efficiently from Gaussian thanks to:
Fattahi, Salar, Richard Y. Zhang, and Somayeh Sojoudi. "Linear-Time Algorithm for Learning Large-Scale Sparse Graphical Models." IEEE Access 7 (2019): 12658-12672.

COMPARISON WITH OTHER METHODS:

We use code for the comparison methods provided by the authors of the following papers:
KPMF: 
Zhou, T., Shan, H., Banerjee, A., & Sapiro, G. (2012, April). Kernelized probabilistic matrix factorization: Exploiting graphs and side information. In Proceedings of the 2012 SIAM international Conference on Data mining (pp. 403-414). Society for Industrial and Applied Mathematics.
GRALS:
Rao, N., Yu, H. F., Ravikumar, P. K., & Dhillon, I. S. (2015). Collaborative filtering with graph information: Consistency and scalable methods. In Advances in neural information processing systems (pp. 2107-2115).


