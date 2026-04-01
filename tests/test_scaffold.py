"""Phase 1: Verify package is importable and exports are correct."""


def test_package_is_importable():
    import engram_memory

    assert hasattr(engram_memory, "__version__")


def test_version_is_string():
    from engram_memory import __version__

    assert isinstance(__version__, str)
    parts = __version__.split(".")
    assert len(parts) >= 2


def test_config_export():
    from engram_memory import Config

    assert Config is not None
