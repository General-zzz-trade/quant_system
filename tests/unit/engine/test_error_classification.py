"""Test classify_exception covers network and venue errors."""


class TestErrorClassification:
    def test_connection_error_classified_as_io(self):
        from engine.errors import classify_exception, ErrorDomain
        result = classify_exception(ConnectionError("refused"))
        assert result.domain == ErrorDomain.IO

    def test_os_error_classified_as_io(self):
        from engine.errors import classify_exception, ErrorDomain
        result = classify_exception(OSError("No such file"))
        assert result.domain == ErrorDomain.IO

    def test_key_error_classified_as_data(self):
        from engine.errors import classify_exception, ErrorDomain
        result = classify_exception(KeyError("missing"))
        assert result.domain == ErrorDomain.DATA

    def test_timeout_still_io(self):
        from engine.errors import classify_exception, ErrorDomain
        result = classify_exception(TimeoutError("timed out"))
        assert result.domain == ErrorDomain.IO

    def test_unknown_defaults_to_engine(self):
        from engine.errors import classify_exception, ErrorDomain
        result = classify_exception(RuntimeError("weird"))
        assert result.domain == ErrorDomain.ENGINE
