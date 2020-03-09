def nb_logging_init() -> None:
    """
    Does initialize logging framework for Jupyter Notebook (StreamHandler with output to cell output).
    :return:
    """
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    logger.addHandler(ch)
