import logging

from pandas import DataFrame

_logger = logging.getLogger(__name__)


def load_classification_data(location: str, file_format: str) -> DataFrame:
    if file_format == "csv":
        import pandas

        _logger.warning(
            "Loading dataset CSV using `pandas.read_csv()` with default arguments and assumed index"
            " column 0 which may not produce the desired schema. If the schema is not correct, you"
            " can adjust it by modifying the `load_file_as_dataframe()` function in"
            " `steps/ingest.py`"
        )

        data = pandas.read_csv(location)
        data['Price_Change'] = (data['Close'].diff() > 0).astype(int)
        data = data.dropna()
        return data
    else:
        raise NotImplementedError
