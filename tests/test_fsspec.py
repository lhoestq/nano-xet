"""Tests of the ``nxet://`` fsspec filesystem."""

import io

import fsspec
import pytest
from fsspec.tests.abstract import AbstractCopyTests, AbstractFixtures, AbstractGetTests

try:
    from fsspec.tests.abstract.mv import AbstractMvTests
except ImportError:  # older fsspec
    AbstractMvTests = object

from nano_xet import NXetFileSystem, NXetStore


def test_protocol_is_registered():
    assert fsspec.get_filesystem_class("nxet") is NXetFileSystem
    assert "nxet" in fsspec.available_protocols()  # via the fsspec.specs entry point


def test_chained_uri_roundtrip(fs, csv_data):
    uri = f"nxet://data/train.csv::file://{fs.root}"
    with fsspec.open(uri, "wb") as f:
        f.write(csv_data)
    with fsspec.open(uri, "rb") as f:
        assert f.read() == csv_data
    opened = fsspec.open(uri, "rb")
    assert opened.path == "data/train.csv"
    assert isinstance(opened.fs, NXetFileSystem)
    assert opened.fs.unstrip_protocol(opened.path) == uri


def test_url_to_fs(fs, csv_data):
    store_fs, path = fsspec.core.url_to_fs(f"nxet://dir/file.bin::file://{fs.root}")
    assert path == "dir/file.bin"
    assert isinstance(store_fs, NXetFileSystem)
    store_fs.pipe_file(path, csv_data)
    assert store_fs.cat_file(path) == csv_data


def test_requires_an_underlying_filesystem():
    with pytest.raises(ValueError, match="underlying filesystem"):
        NXetFileSystem(fo="")
    with pytest.raises(ValueError, match="underlying filesystem"):
        fsspec.filesystem("nxet", fo="nxet://only/path")


def test_file_object_read(fs, csv_data):
    fs.pipe_file("a/b/data.bin", csv_data)
    with fs.open("a/b/data.bin", "rb") as f:
        assert f.read(10) == csv_data[:10]
        f.seek(len(csv_data) - 4)
        assert f.read() == csv_data[-4:]
        f.seek(1000)
        assert f.readline() == csv_data[1000 : csv_data.index(10, 1000) + 1]
        assert f.tell() == csv_data.index(10, 1000) + 1
        assert f.read(0) == b""
        f.seek(0)
        assert f.read(len(csv_data) + 500) == csv_data
        assert f.read(10) == b""
        assert f.tell() == len(csv_data)
    with fs.open("a/b/data.bin", "rb") as f:
        buffer = bytearray(5)
        assert f.readinto(buffer) == 5
        assert bytes(buffer) == csv_data[:5]
        assert f.size == len(csv_data)


def test_file_object_write(fs, csv_data):
    with fs.open("text.txt", "wb") as f:
        f.write(csv_data[:1000])
        f.write(csv_data[1000:])
    assert fs.cat_file("text.txt") == csv_data
    with fs.open("new.bin", "xb") as f:
        f.write(b"hello")
    with pytest.raises(FileExistsError):
        fs.open("new.bin", "xb")
    # empty file
    with fs.open("empty", "wb"):
        pass
    assert fs.cat_file("empty") == b""


def test_append_mode(fs):
    with fs.open("log.txt", "ab") as f:
        f.write(b"first\n")
    with fs.open("log.txt", "ab") as f:
        f.write(b"second\n")
    assert fs.cat_file("log.txt") == b"first\nsecond\n"


def test_text_mode(fs):
    with fs.open("note.txt", "wt") as f:
        f.write("line1\nline2\n")
    assert fs.cat_file("note.txt") == b"line1\nline2\n"
    with fs.open("note.txt", "r") as f:
        assert f.read() == "line1\nline2\n"
        f.seek(0)
        assert sorted(f) == ["line1\n", "line2\n"]


def test_info_and_listing(fs, csv_data):
    fs.pipe_file("top.txt", b"top")
    fs.pipe_file("a/b/one.csv", csv_data)
    fs.pipe_file("a/b/two.csv", b"two")
    fs.pipe_file("a/c/three.csv", b"three")

    assert sorted(n for n in fs.ls("", detail=False)) == ["a", "top.txt"]
    assert sorted(n for n in fs.ls("a", detail=False)) == ["a/b", "a/c"]
    assert fs.ls("a/b/one.csv", detail=False) == ["a/b/one.csv"]

    entry = fs.info("a/b/one.csv")
    assert entry["type"] == "file"
    assert entry["size"] == len(csv_data)
    assert entry["nchunks"] > 1
    assert fs.info("a")["type"] == "directory"
    assert fs.info("a")["size"] is None

    assert fs.exists("a/b/one.csv")
    assert fs.exists("a/b")
    assert fs.isdir("a")
    assert fs.isfile("a/b/one.csv")
    assert not fs.isdir("a/b/one.csv")
    assert not fs.exists("missing")
    with pytest.raises(FileNotFoundError):
        fs.ls("missing")


def test_walk_find_glob(fs, csv_data):
    for name in ["a/b/one.csv", "a/b/two.csv", "a/c/three.csv", "root.csv"]:
        fs.pipe_file(name, csv_data[:100])
    assert sorted(fs.find("")) == sorted(
        ["a/b/one.csv", "a/b/two.csv", "a/c/three.csv", "root.csv"]
    )
    assert sorted(fs.find("a/b")) == ["a/b/one.csv", "a/b/two.csv"]
    assert sorted(fs.glob("a/b/*.csv")) == ["a/b/one.csv", "a/b/two.csv"]
    assert sorted(fs.glob("**/*.csv")) == sorted(
        ["a/b/one.csv", "a/b/two.csv", "a/c/three.csv", "root.csv"]
    )
    assert sorted(where[0] for where in fs.walk("a")) == ["a", "a/b", "a/c"]
    assert fs.sizes(["root.csv", "a/b/one.csv"]) == [100, 100]


def test_delete(fs, csv_data):
    fs.pipe_file("a/b/one.csv", csv_data)
    fs.pipe_file("a/b/two.csv", b"two")
    fs.pipe_file("a/c/three.csv", b"three")

    fs.rm("a/b/one.csv")
    assert not fs.exists("a/b/one.csv")
    assert fs.exists("a/b/two.csv")

    with pytest.raises(IOError, match="recursive"):
        fs.rm("a/c")
    fs.rm("a/c", recursive=True)
    assert not fs.exists("a/c")
    fs.rm("a/b", recursive=True)
    assert fs.ls("", detail=False) == []
    assert fs.store.list_files() == []
    with pytest.raises(FileNotFoundError):
        fs.rm("a/b", recursive=True)


def test_copy_move_and_dedup(fs, csv_data):
    fs.pipe_file("original.csv", csv_data)
    stored = fs.store.index.total_stored_bytes()
    fs.copy("original.csv", "copy.csv")
    assert fs.cat_file("copy.csv") == csv_data
    # a copy inside the store is pure metadata: no new chunk is stored
    assert fs.store.index.total_stored_bytes() == stored
    fs.mv("copy.csv", "moved.csv")
    assert not fs.exists("copy.csv")
    assert fs.cat_file("moved.csv") == csv_data


def test_get_and_put_to_local(fs, tmp_path, csv_data, binary_data):
    fs.pipe_file("dir/remote.bin", binary_data)
    fs.get("dir/remote.bin", str(tmp_path / "out.bin"))
    assert open(tmp_path / "out.bin", "rb").read() == binary_data

    local = tmp_path / "in.bin"
    local.write_bytes(csv_data)
    fs.put(str(local), "uploaded.csv")
    assert fs.cat_file("uploaded.csv") == csv_data
    fs.put(str(local), "uploaded2.csv", recursive=False)
    assert fs.cat_file("uploaded2.csv") == csv_data


def test_dedup_visible_through_the_filesystem(fs, csv_data):
    fs.pipe_file("a", csv_data)
    fs.pipe_file("b", csv_data)
    stats = fs.stats()
    assert stats.files == 2
    assert stats.logical_bytes == 2 * len(csv_data)
    assert stats.stored_bytes == len(csv_data)
    assert stats.dedup_ratio == pytest.approx(0.5, abs=1e-9)
    assert len(fs.xorbs()) < stats.chunks
    assert all(x["nchunks"] > 1 for x in fs.xorbs())


def test_store_survives_a_reopen(fs, store_path, csv_data):
    fs.pipe_file("keep.csv", csv_data)
    other = NXetFileSystem(fo=str(store_path), target_protocol="file")
    assert other.cat_file("keep.csv") == csv_data
    other.pipe_file("new.csv", b"new")
    fs.reload()
    assert fs.cat_file("new.csv") == b"new"
    assert fs.exists("new.csv")


def test_gc_and_head(fs, csv_data, binary_data):
    fs.pipe_file("keep", csv_data)
    fs.pipe_file("drop", binary_data)
    before = {x["name"] for x in fs.xorbs()}
    fs.rm("drop")
    removed = fs.gc()
    assert removed and set(removed) <= before
    assert {x["name"] for x in fs.xorbs()} == before - set(removed)
    for name in removed:
        assert not fs.underlying_fs.exists(fs.store.index.xorb_path(name))
    assert fs.cat_file("keep") == csv_data
    assert fs.header()["chunking"]["min"] == 8192
    assert fs.header()["hash"] == "blake2b-256"


def test_repr_and_urls(fs):
    assert fs.unstrip_protocol("a/b") == f"nxet://a/b::file://{fs.root}"
    assert repr(fs).startswith("NXetFileSystem(nxet://")


def test_dircache_is_invalidated(fs):
    fs.pipe_file("dir/file.txt", b"x")
    assert "dir/file.txt" in fs.ls("dir", detail=False)
    fs.rm("dir/file.txt")
    assert fs.exists("dir/file.txt") is False


def test_local_file_like_usage(fs, binary_data):
    """Behaves like a normal file for a caller that just reads it."""
    fs.pipe_file("data.bin", binary_data)
    with fs.open("data.bin", "rb") as f:
        buffer = io.BytesIO(f.read())
    assert buffer.getvalue() == binary_data
    assert f.closed


def test_store_and_filesystem_share_dedup(store_path, csv_data):
    store = NXetStore.open(f"file://{store_path}")
    store.write_file("from-store.csv", csv_data)
    fs = NXetFileSystem(fo=str(store_path), target_protocol="file")
    assert fs.cat_file("from-store.csv") == csv_data
    assert fs.info("from-store.csv").get("nchunks") == store.info("from-store.csv").nchunks
    store.close()


# -- fsspec's own abstract implementation tests ----------------------------


class TestNXetAbstract(AbstractFixtures):
    """get/copy semantics as defined by fsspec itself."""

    @pytest.fixture
    def fs(self, store_path):
        return NXetFileSystem(fo=str(store_path), target_protocol="file")

    @pytest.fixture
    def fs_path(self, store_path):
        return ""

    @pytest.fixture
    def supports_empty_directories(self):
        """nano-xet directories are virtual: they exist when a file lives under them."""
        return False


@pytest.mark.usefixtures("fs")
class TestCopy(TestNXetAbstract, AbstractCopyTests):
    pass


class TestGet(TestNXetAbstract, AbstractGetTests):
    pass


@pytest.mark.skipif(AbstractMvTests is object, reason="needs fsspec >= 2023.6")
class TestMv(TestNXetAbstract, AbstractMvTests):
    pass


def test_two_spellings_of_one_store_share_state(store_path, csv_data):
    """`fsspec.open(chain)` and NXetFileSystem(fo=dir) must see the same index."""
    import fsspec

    with fsspec.open(f"nxet://a.csv::file://{store_path}", "wb") as f:
        f.write(csv_data)

    fs = NXetFileSystem(fo=str(store_path), target_protocol="file")
    assert fs.exists("a.csv")
    assert fs.cat_file("a.csv") == csv_data

    fs.pipe_file("b.csv", csv_data)
    assert sorted(fs.find("")) == ["a.csv", "b.csv"]


def test_index_reloads_when_another_process_writes(store_path, csv_data):
    """Stale in-memory indexes heal themselves when the index files grew."""
    import fsspec

    reader = fsspec.filesystem("nxet", fo=str(store_path), target_protocol="file")
    writer = fsspec.filesystem("nxet", fo=f"nxet://::file://{store_path}", skip_instance_cache=True)
    assert reader.ls("/") == []

    writer.pipe_file("written/later.csv", csv_data)
    assert reader.cat_file("written/later.csv") == csv_data
    assert reader.info("written/later.csv")["size"] == len(csv_data)
