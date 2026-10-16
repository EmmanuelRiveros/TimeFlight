import pytest

from timeflight import assets_log


def test_log_dedupe_and_markdown(tmp_path):
    log = tmp_path / "assets.jsonl"
    kw = dict(name="LTX-2.5", kind="model", source_url="https://huggingface.co/Lightricks/LTX-2.5",
              license="LTX-2.x Community", attribution="Lightricks", log_path=log)
    assets_log.log_asset(**kw)
    assets_log.log_asset(**kw)
    assets = assets_log.read_assets(log)
    assert len(assets) == 1
    assert "LTX-2.5" in assets_log.render_markdown(assets)


@pytest.mark.parametrize("url", ["https://tile.googleapis.com/v1/3dtiles", "https://api.mapbox.com/x",
                                 "https://www.davidrumsey.com/luna/x"])
def test_forbidden_sources(tmp_path, url):
    with pytest.raises(ValueError):
        assets_log.log_asset("x", "image", url, "CC0", "x", log_path=tmp_path / "a.jsonl")


def test_license_required(tmp_path):
    with pytest.raises(ValueError):
        assets_log.log_asset("x", "image", "https://example.org", " ", "x", log_path=tmp_path / "a.jsonl")
