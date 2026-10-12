from automl.exceptions import AutoMLError, ConfigurationError
from automl.logging_config import DEFAULT_LOG_FORMAT, configure_logging, get_logger


def test_domain_exceptions_share_base_type():
    error = ConfigurationError("invalid config")

    assert isinstance(error, AutoMLError)


def test_logging_can_be_configured():
    configure_logging("WARNING")
    logger = get_logger("automl.test")

    assert logger.name == "automl.test"
    assert "%(levelname)s" in DEFAULT_LOG_FORMAT
