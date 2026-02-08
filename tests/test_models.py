def test_version_exists():
    import urbanpulse
    assert hasattr(urbanpulse, "__version__")
