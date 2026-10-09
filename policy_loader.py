import logging


def load_regulation_text(filepath):
    """Read a policy source as UTF-8 text."""
    logging.info("Loading regulation text from: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as source_file:
        return source_file.read()
