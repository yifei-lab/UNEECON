# UNEECON: UNified inferencE of variant Effects and gene CONstraints
## Synopsis
UNEECON is a statistical method for inferring deleterious mutations and constrained genes in human and potentially other species. Unlike most existing variant and gene prioritization methods, UNEECON integrates both variant-level features and gene-level selective constraints to make predictions. UNEECON is developed by Yifei Huang's research group at the Pennsylvania State University (https://yifei-lab.github.io). The UNEECON program is distributed under the 2-Clause BSD License.

## Precomputed UNEECON scores
Precomputed UNEECON variant scores and UNEECON-G gene scores can be downloaded from [here](https://drive.google.com/drive/folders/15VVHeij5sujGz7QAdc-GcBrRCSLsoXLJ?usp=sharing). It is worth noting that, even though the UNEECON program can be used for both non-commercial and commercial purposes, the precomputed scores are for non-commercial uses only due to the restricted licenses of the input features used for training the UNEECON model.

## Requirements

- python 3.6.8
- TensorFlow 1.13.1
- scikit-learn 0.20.2
- numpy
- scipy
- pandas 

## Quick guide

### Input files

UNEECON requires two input files (a training input file and a prediction input file). Both of the two input files must be gzipped, tab-separated files. As implied by their names, the training input file is used to train the UNEECON model, and the prediction input file is used to predict variant effects and gene constraints after training the UNEECON model. The first four columns of the training input file correspond to mutation ID, presence/absence of the mutation, neutral probability of the occurrence of the mutation, and gene ID. The following columns of the training input file correspond to predictive variant features. Similarly, the first two columns of the prediction input file correspond to mutation ID and gene ID. The following columns of the prediction input file correspond to predictive variant features. The mutation ID and gene ID could be any unique identifiers of mutations and genes. Examples of the input files (training_input.tsv.gz and prediction_input.tsv.gz) can be obtained from [here](https://drive.google.com/drive/folders/19c8avHaoklUpdMPOvC9SRfVdE0wgJKXr?usp=sharing). These example files can only be used for non-commercial purposes.

### Training a nonlinear UNEECON model
With the input files, we can run UNEECON with a single command. For example,
```
$ python UNEECON.py -t training_input.tsv.gz -p prediction_input.tsv.gz -s 42 -r 42 -l 0.0001 -o OUTPUT_DIR -u 512 -n 100
```
*-t*: training input file; *-p*: prediction input file; *-s*: random number seed for splitting training data; *-r*: random number seed for weight initialization; *-l*: initial learning rate; *-o*: output directory; *-u*: number of hidden units; *-n*: number of epochs

If multiple sets of hyperparameters are used, we suggest to fix the *-s* argument so that the results are comparable across different runs. After training, predicted UNEECON variant scores (variant_score.tsv) and UNEECON-G gene scores (gene_random_effect.tsv) will be available in the output directory. One of the output files, gene_random_effect.tsv, also includes a column representing estimated gene-level random effects.

### Training a linear UNEECON model
In the previous example, we used 512 hidden units in the UNEECON model. Alternatively, we could train UNEECON without hidden units. It can be done by calling UNEECON without the *-u* argument. For example,
```
$ python UNEECON.py -t training_input.tsv.gz -p prediction_input.tsv.gz -s 42 -r 42 -l 0.0001 -o OUTPUT_DIR -n 100
```
