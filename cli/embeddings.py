import pathlib

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from cli.printer import pretty_print


USE_URL = 'https://tfhub.dev/google/universal-sentence-encoder/4'

def download_model():
    """Downloads Universal Sentence Encoder model from TFHub"""
    model = hub.load(USE_URL)
    return model


def generate_embeddings(data_out: str, df: pd.DataFrame, model):
    """Generates embeddings for all of the input data"""
    pretty_print('Generating embeddings...')
    titles = df['title'].tolist()
    embeddings = model(titles)
    embeddings = np.array(embeddings).tolist()
    df['embedding'] = embeddings
    df.to_json(data_out, orient='records', lines=True)


def build(data_in: str, data_out: str):
    """Generates embeddings for data and outputs as json"""
    pretty_print('Downloading model...')
    model = download_model()
    df = pd.read_csv(data_in)
    generate_embeddings(data_out, df, model)
    pretty_print('Done. Embeddings successfully created... 🤙')
