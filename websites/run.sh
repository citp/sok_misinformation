#!/bin/bash

# FACT CLASSIFICATION 

## FULL CORPUS
### all features
echo "fact, full corpus, all features"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### no articles
echo "fact, full corpus, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### articles only
echo "fact, full corpus, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove

### no traffic
echo "fact, full corpus, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### traffic only
echo "fact, full corpus, traffic(+)"
python3 train_artifact.py -tk fact -f alexa

### no twitter
echo "fact, full corpus, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### twitter only
echo "fact, full corpus, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified

### no wikipedia
echo "fact, full corpus, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure

### wikipedia only
echo "fact, full corpus, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc


### no url
echo "fact, full corpus, url(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### url only
echo "fact, full corpus, url(+)"
python3 train_artifact.py -tk fact -f url_structure


## MED CORPUS
### all features
echo "fact, med corpus, all features"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### no articles
echo "fact, med corpus, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### articles only
echo "fact, med corpus, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove -sb med

### no traffic
echo "fact, med corpus, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### traffic only
echo "fact, med corpus, traffic(+)"
python3 train_artifact.py -tk fact -f alexa -sb med

### no twitter
echo "fact, med corpus, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### twitter only
echo "fact, med corpus, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb med

### no wikipedia
echo "fact, med corpus, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb med

### wikipedia only
echo "fact, med corpus, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### no url
echo "fact, med corpus, url(-)"
python3 train_artifact.py -tk fact -f alexa, articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### url only
echo "fact, med corpus, url(+)"
python3 train_artifact.py -tk fact -f url_structure -sb med


## SMALL CORPUS
### all features
echo "fact, small corpus, all features"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### no articles
echo "fact, small corpus, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### articles only
echo "fact, small corpus, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove -sb small

### no traffic
echo "fact, small corpus, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### traffic only
echo "fact, small corpus, traffic(+)"
python3 train_artifact.py -tk fact -f alexa -sb small

### no twitter
echo "fact, small corpus, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### twitter only
echo "fact, small corpus, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb small

### no wikipedia
echo "fact, small corpus, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb small

### wikipedia only
echo "fact, small corpus, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### no url
echo "fact, small corpus, url(-)"
python3 train_artifact.py -tk fact -f alexa, articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### url only
echo "fact, small corpus, url(+)"
python3 train_artifact.py -tk fact -f url_structure -sb small


## LEFT BIAS
### all features
echo "fact, left bias, all features"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### no articles
echo "fact, left bias, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### articles only
echo "fact, left bias, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove -sb left

### no traffic
echo "fact, left bias, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### traffic only
echo "fact, left bias, traffic(+)"
python3 train_artifact.py -tk fact -f alexa -sb left

### no twitter
echo "fact, left bias, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### twitter only
echo "fact, left bias, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb left

### no wikipedia
echo "fact, left bias, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb left

### wikipedia only
echo "fact, left bias, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### no url
echo "fact, left bias, url(-)"
python3 train_artifact.py -tk fact -f alexa, articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb left

### url only
echo "fact, left bias, url(+)"
python3 train_artifact.py -tk fact -f url_structure -sb left


## CENTER BIAS
### all features
echo "fact, center bias, all features" 
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### no articles
echo "fact, center bias, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### articles only
echo "fact, center bias, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove -sb center

### no traffic
echo "fact, center bias, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### traffic only
echo "fact, center bias, traffic(+)"
python3 train_artifact.py -tk fact -f alexa -sb center

### no twitter
echo "fact, center bias, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### twitter only
echo "fact, center bias, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb center

### no wikipedia
echo "fact, center bias, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb center

### wikipedia only
echo "fact, center bias, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### no url
echo "fact, center bias, url(-)"
python3 train_artifact.py -tk fact -f alexa, articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb center

### url only
echo "fact, center bias, url(+)"
python3 train_artifact.py -tk fact -f url_structure -sb center


## RIGHT BIAS
### all features
echo "fact, right bias, all features" 
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### no articles
echo "fact, right bias, articles(-)"
python3 train_artifact.py -tk fact -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### articles only
echo "fact, right bias, articles(+)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove -sb right

### no traffic
echo "fact, right bias, traffic(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### traffic only
echo "fact, right bias, traffic(+)"
python3 train_artifact.py -tk fact -f alexa -sb right

### no twitter
echo "fact, right bias, twitter(-)"
python3 train_artifact.py -tk fact -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### twitter only
echo "fact, right bias, twitter(+)"
python3 train_artifact.py -tk fact -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb right

### no wikipedia
echo "fact, right bias, wikipedia(-)"
python3 train_artifact.py -tk fact -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb right

### wikipedia only
echo "fact, right bias, wikipedia(+)"
python3 train_artifact.py -tk fact -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### no url
echo "fact, right bias, url(-)"
python3 train_artifact.py -tk fact -f alexa, articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb right

### url only
echo "fact, right bias, url(+)"
python3 train_artifact.py -tk fact -f url_structure -sb right


#-------------------------------------------------------------
# BIAS CLASSIFICATION

## FULL CORPUS
### all features
echo "bias, full corpus, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### no articles
echo "bias, full corpus, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### articles only
echo "bias, full corpus, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove

### no traffic
echo "bias, full corpus, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### traffic only
echo "bias, full corpus, traffic(+)"
python3 train_artifact.py -tk bias -f alexa

### no twitter
echo "bias, full corpus, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### twitter only
echo "bias, full corpus, traffic(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified

### no wikipedia
echo "bias, full corpus, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure

### wikipedia only
echo "bias, full corpus, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### no url
echo "bias, full corpus, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc

### url only
echo "bias, full corpus, url(+)"
python3 train_artifact.py -tk bias -f url_structure


## MED CORPUS
### all features
echo "bias, med corpus, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### no articles
echo "bias, med corpus, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### articles only
echo "bias, med corpus, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove -sb med

### no traffic
echo "bias, med corpus, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### traffic only
echo "bias, med corpus, traffic(+)"
python3 train_artifact.py -tk bias -f alexa -sb med

### no twitter
echo "bias, med corpus, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### twitter only
echo "bias, med corpus, twitter(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb med

### no wikipedia
echo "bias, med corpus, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb med

### wikipedia only
echo "bias, med corpus, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### no url
echo "bias, med corpus, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb med

### url only
echo "bias, med corpus, url(+)"
python3 train_artifact.py -tk bias -f url_structure -sb med


## SMALL CORPUS
### all features
echo "bias, small corpus, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### no articles
echo "bias, small corpus, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### articles only
echo "bias, small corpus, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove -sb small

### no traffic
echo "bias, small corpus, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### traffic only
echo "bias, small corpus, traffic(+)"
python3 train_artifact.py -tk bias -f alexa -sb small

### no twitter
echo "bias, small corpus, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### twitter only
echo "bias, small corpus, twitter(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb small

### no wikipedia
echo "bias, small corpus, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb small

### wikipedia only
echo "bias, small corpus, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### no url
echo "bias, small corpus, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb small

### url only
echo "bias, small corpus, url(+)"
python3 train_artifact.py -tk bias -f url_structure -sb small


## LOW CREDIBILITY
### all features
echo "bias, low cred, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### no articles
echo "bias, low cred, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### articles only
echo "bias, low cred, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove -sb low

### no traffic
echo "bias, low cred, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### traffic only
echo "bias, low cred, traffic(+)"
python3 train_artifact.py -tk bias -f alexa -sb low

### no twitter
echo "bias, low cred, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### twitter only
echo "bias, low cred, twitter(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb low

### no wikipedia
echo "bias, low cred, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb low

### wikipedia only
echo "bias, low cred, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### no url
echo "bias, low cred, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb low

### url only
echo "bias, low cred, url(+)"
python3 train_artifact.py -tk bias -f url_structure -sb low


## MIXED CREDIBILITY
### all features
echo "bias, mixed cred, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### no articles
echo "bias, mixed cred, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### articles only
echo "bias, mixed cred, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove -sb mixed

### no traffic
echo "bias, mixed cred, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### traffic only
echo "bias, mixed cred, traffic(+)"
python3 train_artifact.py -tk bias -f alexa -sb mixed

### no twitter
echo "bias, mixed cred, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### twitter only
echo "bias, mixed cred, twitter(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb mixed

### no wikipedia
echo "bias, mixed cred, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb mixed

### wikipedia only
echo "bias, mixed cred, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### no url
echo "bias, mixed cred, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb mixed

### url only
echo "bias, mixed cred, url(+)"
python3 train_artifact.py -tk bias -f url_structure -sb mixed


## HIGH CREDIBILITY
### all features
echo "bias, high cred, all features"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### no articles
echo "bias, high cred, articles(-)"
python3 train_artifact.py -tk bias -f alexa,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### articles only
echo "bias, high cred, articles(+)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove -sb high

### no traffic
echo "bias, high cred, traffic(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### traffic only
echo "bias, high cred, traffic(+)"
python3 train_artifact.py -tk bias -f alexa -sb high

### no twitter
echo "bias, high cred, twitter(-)"
python3 train_artifact.py -tk bias -f articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### twitter only
echo "bias, high cred, twitter(+)"
python3 train_artifact.py -tk bias -f twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified -sb high

### no wikipedia
echo "bias, high cred, wikipedia(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure -sb high

### wikipedia only
echo "bias, high cred, wikipedia(+)"
python3 train_artifact.py -tk bias -f wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### no url
echo "bias, high cred, url(-)"
python3 train_artifact.py -tk bias -f alexa,articles_body_glove,articles_title_glove,has_twitter,has_wikipedia,twitter_created_at,twitter_description,twitter_engagement,twitter_haslocation,twitter_urlmatch,twitter_verified,url_structure,wikipedia_categories,wikipedia_content,wikipedia_summary,wikipedia_toc -sb high

### url only
echo "bias, high cred, url(+)"
python3 train_artifact.py -tk bias -f url_structure -sb high
