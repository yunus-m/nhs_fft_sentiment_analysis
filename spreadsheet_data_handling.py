import pandas as pd
import numpy as np

#Add pxtextmining/ to path
import sys
if '../' not in sys.path:
    sys.path.append('../')

from pxtextmining.pxtextmining import params

#%% Dict mapping each minor category to its major category
minor_to_major_cat_dict = {}
for minor_cat in params.minor_cats:
    minor_to_major_cat_dict[minor_cat] = [k for k, v in params.major_cat_dict.items() if minor_cat in v][0]

#%% Helper functions

def cols_to_snake_case(df):
    """Change column names to snake_case
    
    Parameters
    ----------
    df : DataFrame
        df whose columns will be formatted
    
    Returns
    ----------
    DataFrame with snake-case column names
    """
    return df.set_axis(
        df.columns.map(lambda col_name: col_name.lower().strip().replace(' ', '_')),
        axis=1
    )

def clean_text(text):
    """Lightly process text to handle redundant codes.
    
    Parameters
    ----------
    text : str
        Passage of text to process
    
    Returns
    ----------
        Formatted text with some special punctuation mapped to standard equivalent.
    """
    replaced = (
        text.strip()
        .replace('\x0b', ' ')
        .replace('\n', ' ')
        .replace('¬', ' ')
        .replace('Â', '')
        .replace('Ã', '')
        .replace('â', '')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("‚", ",")
        .replace('“', '"')
        .replace('”', '"')
        .replace('„', '"')
        .replace('…', '...')
        .replace("™", 'tm')
        .replace('|🏻', '')
        # .lower() #Option in CountVectorizer. HF tokenizer will handle as relevant.
    )
    return ' '.join(replaced.split())

#%% Tweaking the df
#NB. Dates aren't consistently recorded. Sometimes d/m is swapped.

def tweak_generic(df):
    """Tweak FFT spreadsheet, including question type categorisation.

    -Answer is lightly cleaned
    -Question type column added
    -Answer length columns added
    -Drop null answers

    Parameters
    ----------
    df : DataFrame
        df to format
    
    Returns
    ----------
    Tweaked DataFrame
    """
    return (
        df

        #FFT question -> {nonspecific, could_improve, what_good}
        .assign(question_type=lambda df_: df_['FFT question'].map(params.q_map))

        #Clean answer. Light touch atm, might discard if using llm embeddings.
        .assign(answer_clean=lambda df_: df_['FFT answer'].fillna('').astype(str).map(clean_text))

        #Record length of the answer
        .assign(answer_word_len=lambda df_: df_['answer_clean'].str.split().str.len().astype(int))
        .assign(answer_char_len=lambda df_: df_['answer_clean'].str.len().astype(int))

        #Need text, so drop where answer_lengths==0
        .loc[lambda df_: df_['answer_char_len'].gt(0)]

        #Dates aren't formatted consistently. d/m sometimes swapped.
        # .assign(Date=lambda df_: pd.to_datetime(df_['Date'], format='%d/%m/%Y', errors='coerce'))
    )


def tweak_for_sentiment(df):
    """Tweak FFT data for sentiment analysis
    
    -Generic tweak
    -Drop null sentiment entries
    -Add sentiment score description
    -Convert sentiment score to int
    -Drop columns not used for sentiment analysis

    Parameters
    ----------
    df : DataFrame
        df to tweak for sentiment analysis
    
    Returns
    ----------
    Tweaked df filtered to sentiment-relevant records
    """
    return (
        df
        .pipe(tweak_generic)

        #Drop rows where `Comment sentiment` is NaN
        .loc[lambda df_: df_['Comment sentiment'].notna()]

        #Sentiment textual description from params.sentiment_dict
        .assign(sentiment_desc=lambda df_: df_['Comment sentiment'].map(params.sentiment_dict))

        .astype({'Comment sentiment': int})

        #Select and order columns
        [['Comment ID', 'Trust', 'Respondent ID', 'Date',
          'question_type', 'answer_clean',
          'answer_char_len', 'answer_word_len',
          'Comment sentiment', 'sentiment_desc',
        ]]
    )


def tweak_for_categorisation(df, raw_cat_labels):
    """Tweak FFT data for category tagging analysis
    
    -Generic tweak
    -Drop null category entries
    -Add columns identifying the major and minor tags, and number of tags
    -Drop columns not used for category analysis

    Parameters
    ----------
    df : DataFrame
        df to tweak for category tagging analysis
    
    Returns
    ----------
    Tweaked df filtered to categorisation-relevant records
    """
    return (
        df
        .pipe(tweak_generic)

        #Drop rows that have category "Not assigned"
        .loc[lambda df_: df_['Not assigned'].isna()]
        #Drop categories in the CSV that are not present in params.minor_cats
        .drop(columns=[col for col in raw_cat_labels if col not in params.minor_cats])

        #Summarise all the minor categories for this row into a single column
        # Only consider valid categories. Empty results will be dropped.
        .assign(
            minor_categories=lambda df_:
            df_
            .filter(items=params.minor_cats, axis=1)
            .apply(lambda row_: row_.dropna().index.tolist(), axis=1)
        )
        #Drop rows that didn't have a valid category
        .loc[lambda df_: df_['minor_categories'].map(len).gt(0)]
        #number of minor categories assigned
        .assign(n_minor_categories=lambda df_: df_['minor_categories'].map(len).astype(int))

        #column for major categories (one per minor cat)
        .assign(major_categories=lambda df_: df_['minor_categories'].apply(
            lambda labels_: [minor_to_major_cat_dict[label] for label in labels_],
        ))
        #number of unique major categories
        .assign(
            n_unique_major_categories=lambda df_:
            df_['major_categories'].map(set).map(len).astype(int)
        )

        #Retain columns
        [['Comment ID', 'Trust', 'Respondent ID', 'Date',
          #'Service type 1', 'Service type 2',
          'question_type',  'answer_clean',
          'answer_char_len', 'answer_word_len',
          # 'FFT categorical answer', 'FFT question', 
          # 'Person identifiable info?',

          'minor_categories', 'n_minor_categories',
          'major_categories', 'n_unique_major_categories'
        ]]
    )