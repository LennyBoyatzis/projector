import typer

from cli.embeddings import build as build_embeddings
from cli.projection import build as build_projection


app = typer.Typer()


@app.command()
def embeddings(data_in: str = './data/articles.csv',
                     data_out:str = './data/articles.json'):
    build_embeddings(data_in, data_out)


@app.command()
def projection(data_in: str = './data/articles.json',
               data_out: str = './logs/'):
    build_projection(data_in, data_out)


if __name__ == '__main__':
    app()
