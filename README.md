# LMM-lasso

An implementation of the linear model with sparsity constraints (Lasso model) for association mapping and phenotype prediction, which corrects for population strucure, as described in:

* B. Rakitsch, C. Lippert, O. Stegle, K. Borgwardt (2013) 
**A Lasso Multi-Marker Mixed Model for Association Mapping with Population Structure Correction**,
_Bioinformatics_ 29(2):206-1 [link](http://bioinformatics.oxfordjournals.org/content/29/2/206)

## Usage

The folder `code` contains several Python scripts:
* `lmm_lasso.py` : contains the implementation of the proposed model   
* `test.py` : an example script  

Ina terminal, 

```
python code/test.py
```

The plots are saved in a separate folder named `plots`.


## Data


The folder `code` contains a subfolder `data` which contains:
* `poppheno.csv`  
* `genotypes.csv`  

The data contains 1000 randomly sampled SNPs from the genotype data of A.thaliana from [Atwell et al.(2010)](http://www.nature.com/nature/journal/v465/n7298/full/nature08800.html). For simulating population-driven effects, we used the real phenotype leaf number at flowering time. Univariate analyses as done in the original paper have shown that the phenotype has an excess of associations when population structure is not accounted for. After correction, the p-values are approximately uniformly distributed.

The complete data is public available and can be downloaded at:
[https://cynin.gmi.oeaw.ac.at/home/resources/atpolydb](https://cynin.gmi.oeaw.ac.at/home/resources/atpolydb)




