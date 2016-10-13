This folder contains python scripts for computing the LMM-Lasso:

The folder data contains:
- poppheno.csv
- genotypes.csv

The data contains 1000 randomly sampled SNPs from the genotype data of A.thaliana from Atwell et al.(2010). For simulating population-driven effects, we used the real phenotype leaf number at flowering time. Univariate analyses as done in the original paper have shown that the phenotype has an excess of associations when population structure is not accounted for. After correction, the p-values are approximately uniformly distributed.

The complete data is public available and can be downloaded at:
https://cynin.gmi.oeaw.ac.at/home/resources/atpolydb



The folder pysrc contains
- lmm_lasso.py
- test.py

The file lmm_lasso.py contains the implementation of the proposed model. 
For testing the implementation, go to the directory lmm_lasso and type in:

python pysrc/test.py

Plots are saved in a separate folder plots.



Reference:
Barbara Rakitsch, Christoph Lippert, Oliver Stegle, Karsten Borgwardt
A Lasso Multi-Marker Mixed Model for Association Mapping with Population Structure Correction, Bioinformatics (2012).
