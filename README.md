# IndexFinder
This repository contains a simple algorithm for creating diagnostic indexes on small metabolomic datasets.

This program is based on Noguchi Y. et al investigation [https://academic.oup.com/ajcn/article/83/2/513S/4650434?login=false]. 

**Original algorithm**
The basic algorithm to generate a diagnostic index was as follows: 1) Amino acids are classified into 2 groups on the basis of their correlation with the target parameter, whether continuous, ordinal, or dichotomous, that represents the disease stage or physiologic condition. This gives to a group *P* that contains *Np* amino acids *Aα* (1 ≤ α ≤ Np) having a positive correlation with the target variable, and to a group *Q* that contains *Nq* amino acids *Bβ* (1 ≤ β ≤ Nq) having a negative correlation with the target variable. 2) The second step is to calculate all possible combinations of the variables in the form of the fractional function *F*:

![image](https://user-images.githubusercontent.com/90495911/194709627-fd316a6c-0c0c-494b-abcc-de5fca8ba180.png)

where *ΣAα* is a partial sum of *Aα* selected from *P*, and *ΣBβ* is a partial sum of *Bb* selected from *Q*, with no redundant use of the variables in the function, and where r is the number of fractions in the function. The fractional functions are selected for further processing by evaluation methods such as the sum of squares due to error based on the simple linear regression of F for the continuous target parameter or the variance ratio for the categorical target parameter. 3) The optimal fractional function F, or amino index, is finally selected after cross-validation of a large set of candidate functions by using random sampling to attain robustness with respect to sample variability and noise [https://academic.oup.com/ajcn/article/83/2/513S/4650434?login=false].

**IndexFinder**
Here an attempt is made to implement and improve this algorithm. The program is implemented as a Python module. An IndexFinder class has been created that performs the actions of the algorithm. Included is a Jupyter notebook with usage examples.
