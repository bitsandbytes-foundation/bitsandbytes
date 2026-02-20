import logging
import textwrap

HEADER_WIDTH = 60

logger = logging.getLogger(__name__)


def print_header(txt: str, width: int = HEADER_WIDTH, filler: str = "=") -> None:
    txt = f" {txt} " if txt else ""
    logger.info(txt.center(width, filler))


def print_dedented(text):
    logger.info("\n".join(textwrap.dedent(text).strip().split("\n")))
