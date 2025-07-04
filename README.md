### Supplementary and artifact information for the paper 'SoK: Machine Learning for Misinformation Detection'

_The following instructions identify primary replication results in the prepublication manuscript and provide guidance for reproducing these results on your own machine._ 

For article-scoped replication data files, download the `articles_data` archive [here](https://drive.google.com/file/d/1h8ML2LS8g44M2WpyX2L7Lr_bVMHECftL/view?usp=sharing) (GDrive link; 180.5 MB). 

In this Git repo, you'll see the following three subdirectories: articles, users, and websites. These correspond to the replications found in Sections 5.1, 5.2, and 5.3, respectively, of the manuscript. Instructions for replicating results for each section follow. (These instructions have also been copied into the subdirectory for each scope.)

#### Articles 
See Table 1 in the main paper for the results of replication analyses of the ISOT, FA-KES, and NYTimes and Reuters datasets. Follow instructions in the `analysis` notebook to download and move all relevant data files to the correct location. Execute `./run.sh` to run 30 iterations of the model (with prespecified random seeds) on 1) ISOT and FAKES datasets used in the [original paper](https://www.researchgate.net/publication/348379370_Fake_news_detection_A_hybrid_CNN-RNN_based_deep_learning_approach) and 2) original and modified datasets we developed for purposes of robustness testing. With multithreading (four cores, each running a separate instance of the model with the same seed and a different dataset), this script runs in about 7.5 hours. 

Once complete, `run.sh` will output four files named `final_metrics_{dataset}.txt`, where `dataset` will be one of `{isot, fakes, nyt, reu}`. These files contain the console logs for each round of training and testing. The NYT and REU metrics files will contain performance statistics for original _and_ modified versions of those datasets. We've provided copies of the metrics files in the `outputs_and_analysis` directory of the `articles` directory for your reference. 

A Jupyter notebook, `analysis.ipynb`, is provided in `outputs_and_analysis` for automated extraction and analysis of the classification results reported in the `final_metrics` datafiles. Run all cells in order to reproduce the summary statistics reported in Table 2. __n.b.__: by the labeling convention of the FAKES and ISOT datasets used in the original paper, a _positive_ detection result (i.e., `label = 1`) denotes a text excerpt that has been classified as _true_. Elsewhere in our work, the inverse is true: a positive detection result denotes a `false/misinformative` detection result and a negative detection result denotes a `true` detection result. As such, false positive and negative rates, as reported in our paper, are inverted with respect to what you'll see in the Jupyter notebook: FPRs in the articles-level replication section of the paper are FNRs in the notebook and vice versa. 


#### Users 
See Figure 4 for the results of a partial dependence analysis on TrollMagnifier data. The original paper can be found [here](https://arxiv.org/pdf/2112.00443). The Jupyter notebook containing code to train and run the classifier and create these PDP plots is `trollmagnifier_pdp_artifact.ipynb`. Instructions for use are included in the notebook.


#### Websites
See Table 3 in Appendix D for the results of ablation analyses on an EMNLP dataset for [this paper](https://aclanthology.org/D18-1389.pdf). To run all tests, execute the bashscript contained in the directory using `./run.sh`.

A text header appears at the start of each test case (e.g. "fact, full corpus, all features") denoting 1) the classification task (factuality or bias classification), 2) the corpus to be used (these vary by size (full, medium, small), or are stratified by bias (left, center, right), or credibility (low, mixed, high)), and 3) the feature set. Feature sets are printed to the console at runtime. Extraction code for features is included in the features directory. You can additionally inspect feature sets and function calls in the `run.sh` bashscript. 

Accuracy, MAE, and F1 scores are reported for each test. To compare these results to those found in Table 3, read each table row horizontally, matching accuracy scores to those reported on the console. Hold-one-out analyses are denoted by (-) in the feature set labels (e.g., "articles(-)" denotes _all_ features excluding those relating to article contents and title). Evaluation of specific feature sets are denoted by (+) (e.g., "articles" denotes only those features derived from article contents and title). With the same hardware specs reported previously, the full script runs in about 70 minutes. 

   

