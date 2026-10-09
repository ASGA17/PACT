import logging
import sys
import time
from pathlib import Path
from typing import Optional


def setup_logging(
    task_name: str,
    log_level: int = logging.INFO,
    log_dir: str = "logs",
    run_id: Optional[str] = None,
) -> str:
    """
    Configure project-wide logging.

    Logs are written to both:
    - console (stdout)
    - file: logs/<task_name>/<timestamp>[_<run_id>].log

    This function should normally be called once at the program entry point.

    Returns:
        Absolute path of the created log file.
    """

    task_log_dir = Path(log_dir) / task_name
    task_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if run_id:
        filename = f"{timestamp}_{run_id}.log"
    else:
        filename = f"{timestamp}.log"

    log_file = task_log_dir / filename

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(
        log_file,
        mode="a",
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            file_handler,
            console_handler,
        ],
        force=True,
    )

    return str(log_file.resolve())


if __name__ == "__main__":
    log_path = setup_logging(task_name="test")

    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")
    logger.info("Log file: %s", log_path)
