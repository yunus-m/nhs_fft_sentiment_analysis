
#%%
import torch
import numpy as np

def save_embedding_weights(embedding_module, checkpoint):
    """Save word and positional embeddings from a transformer

    Parameters
    ----------
    embedding_module : torch.nn.Module
        The embedding block with word and position embedding matrices
    checkpoint : str
        Model saved in ./embedding_weights/ to extract weights from

    Returns
    ----------
    None
    """
    pathname_pattern = "./embedding_weights/*_embedding_weights[%s].npy" % checkpoint.replace('/', '--')

    with torch.no_grad():
        word_embedding_weights = embedding_module.word_embeddings.weight.cpu().numpy()
        position_embedding_weights = embedding_module.position_embeddings.weight.cpu().numpy()
    print(word_embedding_weights.shape, position_embedding_weights.shape)

    [
        np.save(pathname_pattern.replace('*', name), weights)
        for name, weights
        in [('word', word_embedding_weights), ('position', position_embedding_weights)]
    ]

def load_embedding_weights(checkpoint):
    """Load word and position embedding weights.

    Parameters
    ----------
    checkpoint : str
        Weights saved in ./embedding_weights/ to load.
    
    Returns
    ----------
    Tuple of np.ndarray (word embedding weights, position embedding weights)
    """
    pathname_pattern = "./embedding_weights/*_embedding_weights[%s].npy" % checkpoint.replace('/', '--')

    return [
        np.load(pathname_pattern.replace('*', name))
        for name in ('word', 'position')
    ]

#%% Discretised version of loguniform dist
from scipy.stats import loguniform

def float_to_int(rvs):
    """Decorator that rounds floats and returns ints
    
    Parameters
    ----------
    rvs : rvs method handle of SciPy distribution

    Returns
    ----------
    Wrapped rvs where the values are rounded integers
    """
    def rvs_wrapper(*args, **kwargs):
        return rvs(*args, **kwargs).round().astype(int)
    return rvs_wrapper

def int_loguniform(low, high):
    """Instantiate a loguniform distribution and patch its .rvs()
    
    Parameters
    ----------
    low, high: upper and lower limits of uniform distribution

    Returns
    ----------
    SciPy loguniform object with integer values upon calling rvs()
    """
    lu = loguniform(float(low), float(high))
    lu.rvs = float_to_int(lu.rvs)
    return lu

#%%
@torch.no_grad()
def compute_metrics(model, loader, device):
    """Compute the class predictions and loss.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch classification model (loaded on `device`)
    loader : torch.utils.data.DataLoader
        Data loader yielding batched samples
    device : str
        Device to map data to for computation

    Returns
    ----------
    tuple of predictions (np.ndarray), CE loss (float)
    """
    model.eval()

    val_preds = []
    cum_loss = 0

    for minibatch in loader:
        minibatch = {k: v.to(device) for k, v in minibatch.items()}
        outputs = model(**minibatch)

        preds = outputs.logits.argmax(dim=1).ravel().cpu()
        val_preds.extend(preds)
        cum_loss += outputs.loss.cpu().numpy() * len(outputs.logits)
    
    return np.array(val_preds), cum_loss / len(loader.dataset)

#%%
import pandas as pd
def pipeline_results_to_df(ans_list, label2id):
    """Formats HuggingFace pipeline() results to a DataFrame, including entropy.

    Parameters
    ----------
    ans_list : list[dict]
        List of a dict per sample, where the dict has results for that sample
    label2id : dict[str, int]
        Mapping of class name to its index

    Returns
    ----------
    DataFrame with a row per sample recording probabilities, entropy, and prediction
    """
    rows = []
    for i, ans in enumerate(ans_list):
        row_df = pd.DataFrame(ans).set_index('labels').T.set_axis([i], axis=0).rename_axis(index='sample')
        rows.append(row_df)
    
    results_df = (
        pd.concat(rows, axis=0)
        [['very positive', 'positive', 'neutral', 'negative', 'very negative']]
        .assign(entropy=lambda df_: df_.apply(lambda row_: -(row_ * np.log2(row_)).sum(), axis=1))
        .assign(prediction=lambda df_: df_.loc[:, 'very positive':'very negative'].idxmax(axis=1).map(label2id))
    )
    return results_df

#%%
from scipy.special import softmax

def logits_to_df(logits, id2label):
    """Format logits from a PyTorch model into a DataFrame, including entropy.

    Parameters
    ----------

    logits : np.ndarray (B, n_classes)
        Model's output logits
    id2label : dict[int, str]
       Mapping of class index to class name

    Returns
    ----------
    DataFrame with a row per sample recording probabilities, entropy, and prediction
    """
    label2id = {v: k for k, v in id2label.items()}
    
    return (
        pd.DataFrame(softmax(logits, axis=1), columns=id2label.values(), index=range(len(logits)))
        .assign(entropy=lambda df_: df_.apply(lambda row_: -np.sum(row_ * np.log2(row_)), axis=1))
        .assign(predicted_label=lambda df_: df_.iloc[:, 0:5].idxmax(axis=1))
        .assign(predicted_id=lambda df_: df_['predicted_label'].map(label2id))
    )

#%%
def combine_question_answer(batch):
    """Combine FFT question and answer into a single sequence
    
    Parameters
    ----------
    batch : HuggingFace Dataset LazyBatch
        dict with a batch of entries for each key
    
    Returns
    ----------
    dict with key "q_and_a" that lists the combined sequence per sample
    """
    question_map = {
        'what_good': 'Question: What was good?',
        'could_improve': 'Question: What could be improved?',
        'nonspecific': 'Feedback:'
    }

    questions = [question_map[q] for q in batch['question_type']]

    answers = [
        ('Answer: ' if question != 'nonspecific' else '') + answer
        for question, answer in
        zip(batch['question_type'], batch['answer_clean'])
    ]

    combined = [q + ' ' + a for q, a in zip(questions, answers)]

    return {'q_and_a': combined}
