from crawl4r.cli.app import app


def test_app_name() -> None:
    assert app.info.name == "crawl4r"
