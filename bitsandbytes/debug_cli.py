import typer

cli = typer.Typer()


@cli.callback()
def callback():
    """
    Awesome Portal Gun
    """


@cli.command()
def shoot():
    """
    Shoot the portal gun
    """
    typer.echo("Shooting portal gun")


@cli.command()
def load():
    """
    Load the portal gun
    """
    typer.echo("Loading portal gun")
