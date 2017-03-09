### Project Description:
This project aims to analyze comments about restaurants on take-out restaurant platforms, including Baidu Nuomi, Meituan, and Eleme.
We use natural language processing and rule-based processing to retrieve comment sentiment, comment themes, keywords, comment dishes,
in order to find out restaurant's detailed superiority and defects in following aspects - service, price, dishes and so on, which could
help them improve themselves with explicit purpose. <br>

### Pipeline:
![yunxiu analysis pipeline flowchart](https://github.com/zhaol07/yunxiu_python_data/blob/master/Comment_NLP_dev/yunxiu_flowchart.png)

### Usage:
**sentiment.py:** main scripts for natural language processing with following modules and functions: <br>
- SentimentScore class: to rate sentiment polarity of comment based on DLUT sentiment thesaurus <br>
- Sentiment.tokenizer: tokenize comment into tokens with pos tag or not <br>
- Sentiment.tf_statistic: statistics of term frequency when training word2vec embedding <br>
- Sentiment.idf_statistic: statistics of term inverse document frequency <br>
- Sentiment.keyword_extraction: extract keywords of input comment based on tf-idf <br>
- Sentiment.normalize_meals: get formal dish name mentioned in comment based on given menu rule <br>
- Sentiment.word2vec_model_train: train word2vec embedding model with given sentences <br>
- Sentiment.lstm_model_train: train lstm classifier with given training set sentences and corresponding labels (convert to word vector first) <br>
- Sentiment.lstm_predict: predict labels of given sentences with pre-trained lstm classifier (convert to word vector first) <br>
- Sentiment.adjust_sentiment: adjust predicted sentiment of sentence whose probability discrepancy between positive prediction and negative prediction is less than 0.2 using rating score as modification rule <br>
- Sentiment.adjust_rating: adjust rating score with prediction sentiment (replace absent rating score with sentiment) <br>
- Sentiment.store_nlp_analysis_results_to_db: save process output dataframe to specified mysql database <br>
- main_total_run function: comment nlp process pipeline <br>
Output contains: **comment_tf.txt, comment_idf.txt** as tf-idf metadata file; **comment_lstm_model.h5, phrase_lstm_model.h5** as lstm model;
**comment_word2vec.model** as word2vec embedding model; **comment_nlp_results, comment_phrase_nlp_results table** in database as pipeline results output <br>

```shell
usage: sentiment.py [-h] [--config CONFIG] [--model_override]
                    [--database_override] [--start_date START_DATE]
                    [--end_date END_DATE]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file with parameters required for
                        process pipeline
  --model_override      Override existing Word2Vec and LSTM model and retrain
                        one
  --database_override   Use existing model to redo prediction of all data in
                        database to override existing output database
  --start_date START_DATE
                        Use data after this date
  --end_date END_DATE   Use data before this date
```

**preprocessing.py**: functions for data input and cleaning <br>
- get_df_from_db: retrieve data from specified database with desired data fields and output a generator returning dataframe <br>
- trad2simp: convert traditional chinese characters in sentence to simplified chinese character <br>
- parse_html_tag: replace html tag remaining from web crawler with ascii character <br>
- mark_entity: replace detailed price/time with symbol "price"/"time" & cantonese word to mandarin word <br>
- remove_redundant_punctuation: remove redundant punctuations <br>
- replace_uncommon_punctuation: substitute/remove uncommon punctuations including emoji like (^.^) <br>
- remove_stop_words: filter out stop words during keywords extraction using tf-idf <br>
- sentence_splitter: split sentence to sub sentences divided by punctuations <br>
- normalize_data_groups: replicate lstm training samples to balance the number of each label <br>

**rule_based_correction.py**: modify prediction results with specified rules <br>
- SentimentCorrection: correct sentiment with given rules and end2end feedback <br>

**feedback_recording.py**: rewrite wrong results with end2end feedback and save feedback to training set <br>
```shell
python feedback_recording.py
```

**ThemeSummarization.py**: <br>
- NameNormalization class: normalize dish names mentioned in comment to their formal name (dish normalization rule table should be given) <br>
- ThemeSummarization class: retrieve themes of 3 levels mentioned in comment based on keyword-theme matching rules (rule table should be given) <br>

**token_clustering.py**: clustering keywords from previous nlp process results to find out potential themes (only manual check now) <br>

**sentiment_analysis_pipeline.sh**: comment nlp process pipeline shell script <br>

**setup.sh**: install required packages for process pipeline in the beginning. Depend on requirement.txt <br>

### Configuration files
**sentiment_config.ini**: configuration parameters for nlp process, mainly in sentiment.py, feedback_recording.py and token_clustering.py scripts <br>

**logging_conf.ini**: configuration of logging recording format <br>

**requirement.txt**: file describing package name and version <br>

### Process dependant materials and rules
**labeling_training_set/comment_label.part2.csv**: training set for training lstm classifier model of whole comment vs sentiment. Same as **comment_training_set** table in comments databse <br>
**labeling_training_set/phrase_label.part2.csv**: training set for training lstm classifier model of phrase (sentence divided by punctuations) vs sentiment. Same as **phrase_training_set** in comments table <br>

**reference/chinese_negation_words.txt**: negation words. Used in sentiment polarity assessment.  Same as **chinese_negation_words** in comments table. <br>
**reference/chinese_stop_words.txt**: stop words used in keywords extraction to filter non-sense words. Same as **chinese_stop_words** table in comments table <br>
**reference/comment_add_dict.txt**: user-defined words used for tokenizing with jieba. Same as **comment_add_dict** table in comments database br>
**reference/comment_del_dict.txt**: words will be suppressed when tokenizing. Same as **comment_del_dict** table in comments database. <br>
**reference/entity_mark.txt**: rules of how to substitute detailed price, detailed time and cantonese words. Same as **entity_mark** table in comments database <br>
**reference/html_tag_replacement**: rules of how to substitute html tags.
**reference/sentiment_polarity_strength.csv**: DLUT sentiment thesaurus, including word, pos, polarity and strength. Same as **sentiment_polarity_strength** table in comments database <br>
**reference/take_away_keyword_rule.level3.experience_keywords.csv**: 3 level keywords to user-experience themes matching rules. Same as **take_away_rule_experience_keywords** in comments databse <br>
**reference/take_away_keyword_rule.level3.process_keyword.csv**: 3 level keywords to delivery process themes matching rules. Same as **take_away_rule_process_keyword** table in comments database <br>
**reference/take_away_keyword_rule.level3.regexp.csv**: use regular expression rules to match themes. Same as **take_away_rule_regexp** table in comments database. <br>
**reference/take_away_keyword_rule.level3.sentiment.csv**: use partial keywords together with sentiment label to match themes. Same as **take_away_rule_sentiment** table in comments database <br>

**reference/menu/branch_store_coreference.txt**: brand store information with enterprise ids. Same as **branch_store_coreference** table in comments database <br>
**reference/menu/xxx.rule.txt**: rules for normalizing dish name to official name. Same as **xxxx_meal** tables in comments database <br>

