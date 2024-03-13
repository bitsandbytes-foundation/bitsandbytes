import textwrap

HEADER_WIDTH = 60


def print_header(txt: str, width: int = HEADER_WIDTH, filler: str = "+") -> None:
    txt = f" {txt} " if txt else ""
    print(txt.center(width, filler))


def print_dedented(text):
    print("\n".join(textwrap.dedent(text).strip().split("\n")))
