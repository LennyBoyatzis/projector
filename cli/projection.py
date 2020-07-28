import os
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

from cli.printer import pretty_print


def save(df: pd.DataFrame, data_out: str):
    """Saves artifacts for tensorboard embeddings projector"""
    df.to_csv(f'{data_out}metadata.tsv', sep='\t', columns=['label', 'title'],
                    index=False, header=True)

    embeddings = df['embedding']
    embeddings = embeddings.tolist()
    embeddings = np.array(embeddings)

    embeddings_tensor = tf.Variable(embeddings, name='embedding') 
    checkpoint = tf.train.Checkpoint(embedding=embeddings_tensor)
    checkpoint.save(os.path.join(data_out, 'embedding.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(data_out, config)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Filters and cleans the dataframe"""
    df_clean = df[df['label'] != 'ios-dev']
    df_clean['title'] = df_clean['title'].apply(lambda x: x.replace('\n', ''))
    return df_clean


def build(data_in: str, data_out: str):
    """Generates artifacts for tensorboard embeddings projector"""
    pretty_print('Creating projection...')
    Path(data_out).mkdir(parents=True, exist_ok=True)
    df = pd.read_json(data_in, orient='records', lines=True)
    df_clean = clean(df)
    save(df_clean, data_out)
    pretty_print('Done. Projection successfully created... 🤙')
