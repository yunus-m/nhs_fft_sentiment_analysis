#%% SentenceEmbedding
from model_utils import load_embedding_weights

from datasets import Dataset
from transformers import AutoTokenizer

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np

class SentenceEmbedding (TransformerMixin, BaseEstimator):
    """Sklearn-compatible transformer that creates sentence embeddings from  text
    
    Parameters
    ----------
    checkpoint : str
        Saved word and/or positional embedding weights to use, saved in embedding_weights/
    positional_encoding : bool
        Additively include positinal encoding (True), otherwise False
    half_precision : bool
        Approximate embeddings as float16; can be useful for regularisation
    batched : bool
        Run HuggingFace tokenizer in batched mode (True), otherwise serially (False)
    """
    # mlm_checkpoint = 'distilbert/distilbert-base-uncased'
    # mlm_checkpoint = 'domain_adapted-distilbert/distilbert-base-uncased'

    # sentence_xformer_checkpoint = 'sentence-transformers/all-mpnet-base-v2'
    # sentiment_checkpoint = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    # sentiment_large_checkpoint = 'TehranNLP-org/bert-large-sst2'
    # !tinybert_checkpoint = 'huawei-noah/TinyBERT_General_4L_312D'

    def __init__(
            self,
            checkpoint='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
            positional_encoding=True,
            half_precision=False,
            batched=True,
    ):
        self.positional_encoding = positional_encoding
        self.half_precision = half_precision
        self.checkpoint = checkpoint
        self.batched = batched
    
    def fit(self, X, y=None):
        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype='object')
        self.n_features_in_ = X.shape[1]

        #Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint.replace('domain_adapted-', ''))
        self.context_window_size = self.tokenizer.model_max_length

        print(f'Records will be trucated to {self.context_window_size} tokens')

        #Load weights
        precision = 'float16' if self.half_precision else 'float32'

        # word embedding weights:     (tokenizer vocab 30_522, emb 768)
        # position embedding weights: (context 512, emb 768)
        self.word_embedding_weights, self.position_embedding_weights = [
            weights.astype(precision) for weights in load_embedding_weights(self.checkpoint)
        ]
        
        if not self.positional_encoding:
            self.position_embedding_weights = np.zeros_like(self.position_embedding_weights)

        return self

    def transform(self, X):
        X_flat_a = np.array(X, dtype=object).ravel()

        dataset = Dataset.from_dict({'answer_clean': X_flat_a})

        if self.batched:
            ids_persample = dataset.map(self._tokenize_to_ids_only, batched=True)['input_ids']
        else:
            ids_persample = [self._text_to_ids(answer)[:self.context_window_size] for answer in dataset['answer_clean']]
        
        sentence_embeddings = np.row_stack([
            (self.word_embedding_weights[sample_tids] + self.position_embedding_weights[:len(sample_tids)]).mean(axis=0)
            for sample_tids in ids_persample
        ])

        return sentence_embeddings
    
    def _text_to_ids(self, text):
        # return self.tokenizer(text, add_special_tokens=False)
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    
    def _tokenize_to_ids_only(self, batch):
        results_list = map(self._text_to_ids, batch['answer_clean'])

        #truncate
        results_list = [tok_answer[:self.context_window_size] for tok_answer in results_list]
        return {'input_ids': results_list}

#%%
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.neighbors import KernelDensity

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """Bayesian generative classification based on KDE
    Copied from: https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html
    
    Parameters
    ----------
    bandwidth : float
        the kernel bandwidth within each class
    kernel : str
        the kernel name, passed to KernelDensity
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel
        
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth,
                                      kernel=self.kernel).fit(Xi)
                        for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0])
                           for Xi in training_sets]
        return self
        
    def predict_proba(self, X):
        logprobs = np.array([model.score_samples(X)
                             for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)
        
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]
    
#%% equivalent to pxtextmining's sklearn sentiment pipeline tuner
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import RandomizedSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from xgboost import XGBClassifier

class PxTextMiningSentimentTunedPipeline (ClassifierMixin, BaseEstimator):
    """Sklearn-compatible estimator with functionality of pxtextmining's sentiment pipeline

    Parameters
    ----------
    model_name : str
        Model to tune. Valid values are "xgbclassifier" (default), "svc"
    n_iter : int
        Number of times hyperparameter space is sampled
    """
    def __init__(self, model_name='xgbclassifier', n_iter=100):
        self.model_name = model_name
        self.n_iter = n_iter

    def fit(self, X, y):
        # X, y = check_X_y(X, y)

        if hasattr(X, 'columns'):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]

        #pxtextmining allows xgb/svc for sentiment
        assert self.model_name in ['svc', 'xgbclassifier'], 'model_name must be {"xgbclassifier", "svc"}'
        self.classes_ = np.unique(y, return_inverse=True)

        preprocessor = make_column_transformer(
            (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['question_type']),
            (TfidfVectorizer(), 'answer_clean'),
            remainder='drop',
            verbose_feature_names_out=False,
        )

        model = (
            XGBClassifier(num_class=len(np.unique(y)), objective='multi:softmax', n_estimators=500, device='cpu')
            if self.model_name == 'xgbclassifier'
            else
            SVC(probability=True, class_weight='balanced', max_iter=1000, cache_size=1000)
        )

        pipeline = make_pipeline(preprocessor, model)

        param_grid_tfidf = {
            'columntransformer__tfidfvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)][0:2], #(2,2) sig. worse
            'columntransformer__tfidfvectorizer__max_df': [0.85, 0.9, 0.95, 0.99],
            'columntransformer__tfidfvectorizer__min_df': [0.0] + list(range(1, 12)), #0 int not allowed
        }

        #We are specfiying the model rather than searching over models.
        param_grid_xgb = {
            'xgbclassifier__max_depth': [4, 5, 6, 7, 8],
            'xgbclassifier__min_child_weight': [0.5, 1, 2, 5],
            'xgbclassifier__gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5],
        }

        param_grid_svc = {
            'svc__C': [1, 5, 10, 15, 20],
            'svc__kernel': ['rbf', 'linear', 'sigmoid'][0:1], #latter 2 sig. worse
        }

        param_grid = param_grid_tfidf | (
            param_grid_xgb if self.model_name == 'xgbclassifier' else param_grid_svc
        )
        
        self.pipeline_ = RandomizedSearchCV(
            pipeline,
            param_grid,
            cv=4,
            scoring='average_precision', #tunes using ap
            n_iter=self.n_iter,
        ).fit(X, y)

        return self
    
    def predict(self, X, *args, **kwargs):
        return self.pipeline_.predict(X, *args, **kwargs)
    def predict_proba(self, X, *args, **kwargs):
        return self.pipeline_.predict_proba(X, *args, **kwargs)
    def score(self, X, y, *args, **kwargs):
        return self.pipeline_.score(X, y, *args, **kwargs)

if False:
    #Test and format results
    pxs = PxTextMiningSentimentTunedPipeline('svc').fit(features_df, labels_ser)
    (
        pd.DataFrame(pxs.cv_results_).drop(columns='params')
        .loc[:, lambda df_: [c for c in df_.columns if not c.startswith('split')]]
        .drop(columns=['std_fit_time', 'std_score_time'])
        .pipe(lambda df_:
            df_.set_axis([c.replace('param_columntransformer__', '...') for c in df_.columns], axis=1)
            )
        .sort_values('rank_test_score')
        .style.bar(subset='mean_test_score', align='left', cmap='Reds')
    )

#%% Since catboost returns (n_samples,1), it fails with voting classifier
from catboost import CatBoostClassifier

class ModifiedCatBoostClassifier(CatBoostClassifier):
    """Wraps CatBoostClassifier to return flattened predictions  (required for sklearn.ensemble)
    """
    def predict(self, data, **kwargs):
        predictions = super().predict(data, **kwargs)
        return predictions.ravel()

#%% Custom MLP
# Supporting class weights

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device, '| GPU name', torch.cuda.get_device_name())

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, compute_class_weight

class CustomMLP(ClassifierMixin, BaseEstimator):
    """Sklearn-compatible MLP classifier supporting class weights
    
    Parameters
    ----------
    hidden_layer_size : int
        Currently just a single hidden layer, with size `hidden_layer_size`
    batch_size : int
        Minibatch size
    class_weight : Union[None, str]
        Class weights - None for no class balancing and "balanced" otherwise
    alpha : float
        Weight decay value
    n_epochs : int
        Number of training epochs
    nesterovs_momentum : bool
        Use AdamW (False) vs NAdam (True)
    device : str
        Device to map the model and data to for computation ("cuda", "cpu")
    """
    def __init__(self, hidden_layer_size=200, batch_size=16, class_weight=None, alpha=0, n_epochs=20, nesterovs_momentum=False, device='cpu'):
        self.hidden_layer_size = hidden_layer_size
        self.class_weight = class_weight
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.nesterovs_momentum = nesterovs_momentum #nb sklearn's version only when sgd
        self.device = device
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_classes_ = np.unique(y).size
        self.classes_ = np.unique(y)

        assert self.class_weight in ['balanced', None], 'Invalid class_weight'
        self.class_weight_ = (
            compute_class_weight('balanced', classes=self.classes_, y=y)
            if self.class_weight == 'balanced'
            else np.ones_like(self.classes_)
        )
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.class_weight_).to(self.device).float())

        loader = DataLoader(
            TensorDataset(torch.tensor(X).float(), torch.tensor(y).long()),
            batch_size = self.batch_size,
            pin_memory=False if self.device == 'cpu' else True,
        )

        self.model_ = nn.Sequential(
            nn.BatchNorm1d(self.n_features_in_),

            nn.Linear(self.n_features_in_, self.hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_layer_size),

            nn.Linear(self.hidden_layer_size, self.n_classes_),
        ).to(self.device)

        optimizer = (torch.optim.NAdam if self.nesterovs_momentum else torch.optim.AdamW)(self.model_.parameters(), weight_decay=self.alpha)

        for _ in range(self.n_epochs):
            self.model_.train()

            for (X_mb, y_mb) in loader:
                X_mb, y_mb = [tens.to(self.device) for tens in (X_mb, y_mb)]
                logits = self.model_(X_mb)
                loss = loss_fn(logits, y_mb)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict(self, X):
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.tensor(X).float().to(self.device))
        return logits.argmax(dim=1).cpu().numpy()
