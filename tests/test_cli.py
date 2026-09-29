import json
import os

import pytest

from nano_xet.cli import main
from nano_xet.store import human_bytes


@pytest.fixture
def store_uri(store_path):
    return f"nxet://::file://{store_path}"


def uri_of(store_path, path=""):
    return f"nxet://{path}::file://{store_path}"


def test_put_ls_cat_get(tmp_path, store_path, store_uri, csv_data, capsys):
    local = tmp_path / "train.csv"
    local.write_bytes(csv_data)

    assert main(["put", str(local), uri_of(store_path, "data/")]) == 0
    assert "data/train.csv" in capsys.readouterr().out

    assert main(["ls", uri_of(store_path, "data")]) == 0
    listing = capsys.readouterr().out
    assert "train.csv" in listing
    assert human_bytes(len(csv_data)) in listing

    assert main(["ls", store_uri, "-R"]) == 0
    assert "data/train.csv" in capsys.readouterr().out

    assert main(["cat", uri_of(store_path, "data/train.csv")]) == 0
    assert capsys.readouterr().out == csv_data.decode()

    out_file = tmp_path / "copy.csv"
    assert main(["get", uri_of(store_path, "data/train.csv"), str(out_file)]) == 0
    assert out_file.read_bytes() == csv_data


def test_put_renames_a_single_file(tmp_path, store_path, store_uri, csv_data, capsys):
    local = tmp_path / "train.csv"
    local.write_bytes(csv_data)
    main(["put", str(local), uri_of(store_path, "renamed.csv")])
    capsys.readouterr()
    main(["ls", store_uri, "-R"])
    assert "renamed.csv" in capsys.readouterr().out
    assert "train.csv" not in capsys.readouterr().out


def test_put_two_files_dedup(tmp_path, store_uri, csv_data, capsys):
    for name in ("a.csv", "b.csv"):
        (tmp_path / name).write_bytes(csv_data)
    main(["put", str(tmp_path / "a.csv"), str(tmp_path / "b.csv"), store_uri])
    capsys.readouterr()
    main(["stats", store_uri])
    stats_out = capsys.readouterr().out
    assert "dedup" in stats_out.lower()
    assert "2" in stats_out


def test_get_recursive(tmp_path, store_path, store_uri, csv_data, capsys):
    for name in ("one.csv", "two.csv"):
        main(["put", str(_file(tmp_path, name, csv_data)), uri_of(store_path, "dir/")])
    capsys.readouterr()
    target = tmp_path / "out"
    assert main(["get", uri_of(store_path, "dir"), str(target), "-r"]) == 0
    assert sorted(p.name for p in target.rglob("*")) == ["one.csv", "two.csv"]

    target2 = tmp_path / "out2"
    assert main(["get", store_uri, str(target2), "-r"]) == 0
    copied = sorted(str(p.relative_to(target2)) for p in target2.rglob("*") if p.is_file())
    assert copied == [f"dir{os.sep}one.csv", f"dir{os.sep}two.csv"]


def test_rm_and_gc(store_path, store_uri, csv_data, binary_data, capsys, tmp_path):
    (tmp_path / "keep.csv").write_bytes(csv_data)
    (tmp_path / "drop.bin").write_bytes(binary_data)
    main(["put", str(tmp_path / "keep.csv"), str(tmp_path / "drop.bin"), store_uri])
    capsys.readouterr()

    main(["xorbs", store_uri])
    before = capsys.readouterr().out
    assert before.strip() and ".xorb" in before

    main(["rm", uri_of(store_path, "drop.bin")])
    assert "deleted" in capsys.readouterr().out

    main(["gc", store_uri, "--dry-run"])
    assert "would delete" in capsys.readouterr().out

    main(["gc", store_uri])
    assert "deleted" in capsys.readouterr().out

    main(["gc", store_uri])
    assert "nothing to delete" in capsys.readouterr().out

    main(["cat", uri_of(store_path, "keep.csv")])
    assert capsys.readouterr().out == csv_data.decode()


def test_stats_and_head(store_path, store_uri, csv_data, capsys, tmp_path):
    (tmp_path / "f.csv").write_bytes(csv_data)
    main(["put", str(tmp_path / "f.csv"), store_uri])
    capsys.readouterr()

    assert main(["stats", store_uri]) == 0
    printed = capsys.readouterr().out
    assert "file(s)" in printed and "xorb(s)" in printed and "dedup" in printed

    assert main(["head", store_uri]) == 0
    header = json.loads(capsys.readouterr().out)
    assert header["format"] == "nano-xet"
    assert header["chunking"]["mean"] == 65536


def test_unknown_file_fails(store_path):
    with pytest.raises(FileNotFoundError):
        main(["cat", uri_of(store_path, "nope.csv")])


def test_missing_store_fails():
    with pytest.raises(ValueError, match="underlying filesystem"):
        main(["ls", "nxet://"])


def _file(directory, name, data):
    path = directory / name
    path.write_bytes(data)
    return path
