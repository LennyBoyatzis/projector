import typer


def pretty_print(msg: str):
    """Prints colored msg to the console"""
    typer.echo(typer.style(msg, fg=typer.colors.GREEN, bold=True))
