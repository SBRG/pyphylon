# Non-negative Matrix Factorization (NMF) in Phylogenetic Analysis

## Overview

Non-negative Matrix Factorization (NMF) is a powerful technique used in pyphylon for 
decomposing complex genomic data into interpretable components. This document provides 
a detailed explanation of NMF and its application in phylogenetic analysis.

## Mathematical Formulation

Given a non-negative matrix V ∈ ℝ^(n×m), NMF aims to find two non-negative matrices 
W ∈ ℝ^(n×k) and H ∈ ℝ^(k×m) such that:

V ≈ WH

The optimization problem is formulated as:

min_{W,H} ||V - WH||²_F
subject to W ≥ 0, H ≥ 0

where ||·||_F denotes the Frobenius norm.

## Interpretation in Phylogenetic Context

In the context of phylogenetic analysis:
- V represents the gene presence/absence matrix across different species or strains
- W represents the basis components, which can be interpreted as "phylons" or ancestral gene groups
- H represents the coefficient matrix, indicating the contribution of each phylon to each species/strain

## Algorithm

pyphylon uses the multiplicative update rules algorithm for NMF:

1. Initialize W and H with non-negative values
2. Iterate until convergence or maximum iterations:
   a. H ← H * (W^T V) / (W^T W H)
   b. W ← W * (V H^T) / (W H H^T)

## Advantages in Phylogenetic Analysis

- Discovers latent structures in genomic data
- Provides interpretable components (phylons)
- Allows for fractional assignment of genes to phylons
- Robust to noise and missing data

## Limitations and Considerations

- Non-unique solutions
- Sensitivity to initialization
- Determining the optimal number of components (ranks)

## References

1. Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative 
   matrix factorization. Nature, 401(6755), 788-791.
2. Ding, C., Li, T., & Peng, W. (2008). On the equivalence between non-negative matrix 
   factorization and probabilistic latent semantic indexing. Computational Statistics 
   & Data Analysis, 52(8), 3913-3927.