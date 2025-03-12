This repo is a collection of notebooks that explore NHS patient experience feedback* with the aim of developing a resource-efficient sentiment analysis model. My experiments build on the [reports and findings](https://the-strategy-unit.github.io/PatientExperience-QDC/) of the [`pxtextmining`](https://github.com/The-Strategy-Unit/pxtextmining/tree/main) project.

*The sample Friends and Family Test (FFT) data is publicly hosted at the [`pxtextmining`](https://github.com/The-Strategy-Unit/pxtextmining/tree/main) project.

I explore and reference the data and code that has been helpfully open-sourced by [The Strategy Unit](https://github.com/The-Strategy-Unit) team based in the NHS. I am not affiliated with the NHS and this repo is a personal side-project.

## Main finding
The main finding of my work is that TinyBERT performs competitively with DistilBERT for sentiment classification of NHS FFT free-text feedback, both in terms of overall performance ($F_1^{macro}$ of 64% vs 65%), and for 'critical'/'negative' reponses (recall of 65% vs 71%).

This finding is relevant to deployment settings where resources are limited, since TinyBERT (15M parameters) is one fifth the size of DistilBERT, resulting in a smaller resource footprint and shorter run times.

## Main code and model outputs

The main code and model outputs of my exploratory work are:
 - TinyBERT fine-tuned for FFT sentiment classification
 - `SentenceEmbedding` custom transformer
   - Converts text to a sentence embedding using word and/or positional embeddings extracted from LLMs without needing to access an LLM
 - The `pxtextmining` sentiment analysis pipeline wrapped as a `scikit-learn`-compatible estimator
   - `PxTextMiningSentimentTunedPipeline` - useful for cross-validation scoring
 - CatBoost + feature engineering pipeline with good general performance
 - `CustomMLP` - a `scikit-learn`-compatible estimator
   - Similar to `MLPClassifier`, but supports class weights which generally work well with the FFT sentiment data
 - DistilBERT domain-adapted to FFT feedback
     - Whereas DistilBERT might predict `movie|story|film` in `The ___ took a while`, the domain-adapted version might predict `appointment|journey|wait`

## Other findings of interest
 - CatBoost with appropriate feature engineering has good overall performance compared with DistilBERT ($F_1^{macro}$ of 59% vs 65%), but its recall for the 'very negative' class of feedback is inferior (38% vs 68%).
 - Hierarchical classification did not afford much improvement for the 'very negative' class
   - Two hierarchical approaches were explored: polarity hierarchy, and sentiment strength hierarchy
 - Threshold tuning the 'very negative' class for higher recall resulted in a sharp drop in precision
   - But for neutral/mixed feedback, we can gain a lot of recall for a minimal trade in precision 


## Repo structure

**`1 - eda.ipynb`**
Exploratory data analysis of the sentiment and category data.

![image](https://github.com/user-attachments/assets/92ac3f79-63d7-48c2-af2b-d8cbbbaf03fc)

**`2 - sentiment_baseline.ipynb`**
Assessing how well NLTK Vader and TextBlob track sentiment labels.

![image](https://github.com/user-attachments/assets/c871f5c4-bb43-4145-b485-13f2dd2af25a)

**`3 - sklearn_pipeline.ipynb`**
Developing a sentiment classification pipeline using traditional ML tooling.

![image](https://github.com/user-attachments/assets/655a1fd5-4395-4274-9a58-d49699e41e7b)

**`4 - tinybert_finetuning.ipynb`**
Finetuning TinyBERT first on SST-5 sentiment data, then further tuning it on FFT data.

![image](https://github.com/user-attachments/assets/aed20549-a668-4f47-9dde-facd7cff9515)

**`5 - domain_adaptation.ipynb`**
Domain adaptation of DistilBERT using masked language modelling (MLM). I originally used this to generate domain-adapted word embeddings when developing a lightweight sentence-embedding scheme. 

![image](https://github.com/user-attachments/assets/c7647a6b-d374-46f1-b347-7024d5128b4c)

**`custom_estimators.py`** Contains various custom estimators, including for sentence embedding and a class-weighted MLP

**`model_utils.py`** Tools for model data handling and processing

**`spreadsheet_data_handling.py`** Loading and light-touch formatting of NHS FFT data

### Environment setup
You can run `pip install -r requirements.txt` in a new `conda` environment to get an identical setup.

Alternatively, for manual installation, the main dependencies are: `catboost`, `lightgbm`, `xgboost`, `pandas`, `scikit-learn`, `torch`, `tensorboard`, `transformers[sentencepiece]`, `seaborn`, `jupyterlab`, `ipykernel`, and `ipympl`.

`scikit-learn-intelex` is helpful if you have an Intel CPU. I am running the CUDA-enabled version of PyTorch.

 ## Further avenues to explore
 - Combine domain adaptation with fine-tuning with the aim of improving TinyBERT performance
 - Cluster analysis of the neutral/mixed responses
   - Being able to separate these out would help with surfacing critical comments and aid model training
 - Apply the methods explored so far to category tagging
   - Categories have been explored in the EDA notebook, but not studied further   
