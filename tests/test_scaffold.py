"""Phase 1: Verify package is importable and exports are correct."""


def test_package_is_importable():
    import engram

    assert hasattr(engram, "__version__")


def test_version_is_string():
    from engram import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


def test_config_export():
    from engram import Config

    assert Config is not None
