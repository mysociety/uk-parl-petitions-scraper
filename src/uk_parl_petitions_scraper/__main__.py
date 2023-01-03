import rich_click as click
from .build import build_database


@click.group()
def cli():
    pass


def main():
    cli()


@cli.command()
def build():
    print("Downloading petitions data based on keywords")
    build_database()


if __name__ == "__main__":
    main()
