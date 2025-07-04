### Supplementary and artifact information for the paper 'SoK: Machine Learning for Misinformation Detection': website-level replication

_The following instructions identify primary replication results in the prepublication manuscript and provide guidance for reproducing these results on your own machine._ 

#### Websites
See Table 3 in Appendix D for the results of ablation analyses on an EMNLP dataset for [this paper](https://aclanthology.org/D18-1389.pdf). To run all tests, execute the bashscript contained in the directory using `./run.sh`.

A text header appears at the start of each test case (e.g. "fact, full corpus, all features") denoting 1) the classification task (factuality or bias classification), 2) the corpus to be used (these vary by size (full, medium, small), or are stratified by bias (left, center, right), or credibility (low, mixed, high)), and 3) the feature set. Feature sets are printed to the console at runtime. Extraction code for features is included in the features directory. You can additionally inspect feature sets and function calls in the `run.sh` bashscript. 

Accuracy, MAE, and F1 scores are reported for each test. To compare these results to those found in Table 3, read each table row horizontally, matching accuracy scores to those reported on the console. Hold-one-out analyses are denoted by (-) in the feature set labels (e.g., "articles(-)" denotes _all_ features excluding those relating to article contents and title). Evaluation of specific feature sets are denoted by (+) (e.g., "articles" denotes only those features derived from article contents and title). With the same hardware specs reported previously, the full script runs in about 70 minutes. 

The README accompanying the original code release for the replicated user-level paper follows below:

# Factuality and Bias Prediction of News Media

This repository describes the work that was published in two papers (see citations below) on predicting the factuality and political bias in news media. Each paper proposes a different set of engineered features collected from sources of information related to the target media.

```
@InProceedings{baly:2018:EMNLP2018,
  author      = {Baly, Ramy  and  Karadzhov, Georgi  and  Alexandrov, Dimitar and  Glass, James  and  Nakov, Preslav},
  title       = {Predicting Factuality of Reporting and Bias of News Media Sources},  
  booktitle   = {Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  series      = {EMNLP~'18},
  NOmonth     = {November},
  year        = {2018},
  address     = {Brussels, Belgium},
  NOpublisher = {Association for Computational Linguistics}
}
```

```
@InProceedings{baly:2020:ACL2020,
  author      = {Baly, Ramy and Karadzhov, Georgi and An, Jisun and Kwak, Haewoon and Dinkov, Yoan and Ali, Ahmed and Glass, James and Nakov, Preslav},
  title       = {What Was Written vs. Who Read It: News Media Profiling Using Text Analysis and Social Media Context},  
  booktitle   = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  series      = {ACL~'20},
  NOmonth     = {July},
  year        = {2020},
  NOpublisher = {Association for Computational Linguistics}
}
```

## Corpus
The corpus was created by retrieving websites along with their factuality and bias labels from the Media Bias/Fact Check (MBFC) [website](http://mediabiasfactcheck.com/).  Two versions of the corpus ("emnlp18" and "acl2020") can be found at `./data/{version}/corpus.tsv`, and contains the following fields:
* **source_url**: the URL to each website (example: http://www.who.int/en/)
* **source_url_normalized**: a shortened version of the *source_url* (example: who.int-en). These will be used as IDs to split the data into 5 folds of training and testing (in `./data/splits.txt`)
* **ref**: the link to the page in the MBFC website analyzing the corresponding website (example: http://mediabiasfactcheck.com/world-health-organization-who/)
* **fact**: the factuality label of each website (low, mixed, or high)
* **bias**: the bias label of each website (extreme-right, right, center-right, center, center-left, left, extreme-left)

## Features
In addition to the corpus, we provide the different features that we used to obtain the results in our papers. We also include the script that reads these features, train the SVM classifier and writes the performance metrics and output predictions to file.  The features can be found at `./data/{version}/features/`.

1. For the *"emnlp18"* paper, the following features are used:
    - articles_body_glove
    - articles_title_glove
	- has_twitter
	- has_wikipedia
	- twitter_created_at
	- twitter_description
	- twitter_engagement
    - twitter_haslocation
	- twitter_urlmatch
	- twitter_verified
	- url_structure
	- wikipedia_categories
	- wikipedia_content
	- wikipedia_summary
	- wikipedia_toc

2. For the *"acl2020"* paper, the following features are used:
	- articles_body_bert
	- articles_title_bert
	- has_facebook
	- has_twitter
	- has_wikipedia
	- has_youtube
	- twitter_profile
	- twitter_followers
	- wikipedia_content
	- youtube_fulltext
	- youtube_nela
	- youtube_numerical
	- youtube_opensmile
    - youtube_subs

Details about each feature can be found in the cited papers. Each of these features is stored as a JSON file, where each key correspond to a source_url (normalized), and its value is a list of numerical values representing this particular feature.

## Training and Classification
To run the training script, use a command-line that follows the template below.

```
python3 train.py -tk [0] -f [1] -ds [2]
```

where
* [0] is the task at hand: "fact" or "bias" prediction
* [1] is the list of features (from the lists above) that will be used to train the model. features must be comma-separated.
* [2] is the name of the dataset we are running the experiment on ("acl2020" or "emnlp18").

The performance metrics and output predictions will be stored in `./data/{version}/results/{task}_{features}/`
