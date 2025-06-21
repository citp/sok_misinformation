### Supplementary and artifact information for the paper 'SoK: Machine Learning for Misinformation Detection'

_The following instructions identify primary replication results in the prepublication manuscript and provide guidance for reproducing these results on your own machine._ 

Download the replications archive [here](https://drive.google.com/file/d/1EaALk1ZswwAxcFXH-HvES4lDsFarMgZ_/view?usp=sharing) (GDrive link; 184 MB). 

On unzipping `replications.zip`, you'll see three directories: articles, users, and websites. These correspond to the replications found in Sections 5.1, 5.2, and 5.3, respectively, of the manuscript. Instructions for replicating results for each section follow. 

#### Articles 
See Table 2 for results of a replication analyses of the ISOT, FA-KES, and NYTimes and Reuters datasets. Call `CNN_RNN.py`  to run the classifier on both datasets&mdash;FA-KES and ISOT&mdash;originally presented in the [paper](https://www.researchgate.net/publication/348379370_Fake_news_detection_A_hybrid_CNN-RNN_based_deep_learning_approach). You'll need to edit the input dataset on line 53 (`dataset = pd.read_csv([filename])`), where `[filename]` can be either `isot.csv` or `fakes.csv`.

To test the classifier on new datasets used in the replication, call `CNN_RNN_nyt_reu.py --test_file [filename]`. Possible filenames are `modified_reuters.csv`, `modified_nytimes.csv`, `real_reuters.csv`, and `real_nytimes.csv`. Each call runs in about 4 minutes on a Macbook with 16 GB of memory and a 2.7 GHz Quad-Core Core i7 processor.


#### Users 
See Figure 4 for the results of a partial dependence analysis on TrollMagnifier data. The original paper can be found [here](https://arxiv.org/pdf/2112.00443). The Jupyter notebook containing code to train and run the classifier and create these PDP plots is `trollmagnifier_pdp_artifact.ipynb`. Instructions for use are included in the notebook.


#### Websites
See Table 4 in Appendix D for the results of ablation analyses on an EMNLP dataset for [this paper](https://aclanthology.org/D18-1389.pdf). To run all tests, execute the bashscript contained in the directory using `./run.sh`.

A text header appears at the start of each test case (e.g. "fact, full corpus, all features") denoting 1) the classification task (factuality or bias classification), 2) the corpus to be used (these vary by size (full, medium, small), or are stratified by bias (left, center, right), or credibility (low, mixed, high)), and 3) the feature set. Feature sets are printed to the console at runtime. Extraction code for features is included in the features directory. You can additionally inspect feature sets and function calls in the `run.sh` bashscript. 

Accuracy, MAE, and F1 scores are reported for each test. To compare these results to those found in Table 4, read each table row horizontally, matching accuracy scores to those reported on the console. Hold-one-out analyses are denoted by (-) in the feature set labels (e.g., "articles(-)" denotes _all_ features excluding those relating to article contents and title). Evaluation of specific feature sets are denoted by (+) (e.g., "articles" denotes only those features derived from article contents and title). With the same hardware specs reported previously, the full script runs in about 70 minutes. 

   

