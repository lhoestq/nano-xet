"""End-to-end tests of the nano-xet storage engine."""

import json

import pytest

from nano_xet import NXetStore
from nano_xet.index import FILES_INDEX_NAME, HEADER_NAME, XORBS_INDEX_NAME
from nano_xet.store import MAX_XORB_BYTES


def test_write_then_read_whole_file(store, csv_data):
    record = store.write_file("data/train.csv", csv_data)
    assert record.size == len(csv_data)
    assert record.nchunks > 1  # enough data for several chunks
    assert store.read_file("data/train.csv") == csv_data
    assert store.list_files() == ["data/train.csv"]


def test_read_ranges(store, csv_data):
    store.write_file("a.txt", csv_data)
    for start, end in [
        (0, len(csv_data)),
        (0, 1),
        (100, 1000),
        (8191, 8193),
        (100_000, 200_000),
        (len(csv_data) - 5, None),
        (len(csv_data), None),
        (len(csv_data) + 10, None),
        (5, 3),
    ]:
        assert store.read_range("a.txt", start, end) == csv_data[start:end], (start, end)


def test_binary_and_empty_files(store, binary_data):
    store.write_file("big.bin", binary_data)
    store.write_file("empty.bin", b"")
    store.write_file("tiny.bin", b"x")
    assert store.read_file("big.bin") == binary_data
    assert store.read_file("empty.bin") == b""
    assert store.read_file("tiny.bin") == b"x"
    assert store.info("empty.bin").size == 0


def test_xorbs_are_bigger_than_chunks(store, csv_data):
    record = store.write_file("train.csv", csv_data)
    xorbs = list(store.index.xorbs.values())
    assert xorbs, "the data must be stored in xorbs"
    assert len(xorbs) < record.nchunks  # several chunks share one physical object
    for xorb in xorbs:
        assert xorb.size == sum(size for _, size, _ in xorb.chunks)
        # chunks are ordered by hash inside a xorb, like in Xet
        hashes = [h for h, _, _ in xorb.chunks]
        assert hashes == sorted(hashes)
        offsets = [off for _, _, off in xorb.chunks]
        assert offsets == sorted(offsets)


def test_files_can_be_rebuilt_from_the_physical_objects(store, csv_data):
    """The index links (file -> chunks -> xorb+offset) are enough to read a file."""
    store.write_file("train.csv", csv_data)
    store.close()  # no cached read handles
    pieces = []
    for hash_hex, size in store.info("train.csv").chunks:
        location = store.index.chunks[hash_hex]
        with open(store.index.xorb_path(location.xorb), "rb") as f:
            f.seek(location.offset)
            chunk = f.read(size)
        assert len(chunk) == size
        pieces.append(chunk)
    assert b"".join(pieces) == csv_data


def test_deduplication_of_identical_files(store, csv_data):
    store.write_file("first.csv", csv_data)
    stored_after_first = store.index.total_stored_bytes()
    store.write_file("second.csv", csv_data)
    assert store.index.total_stored_bytes() == stored_after_first  # nothing new stored
    assert store.last_write["dedup_chunks"] == store.last_write["chunks"]
    assert store.last_write["new_bytes"] == 0
    assert store.read_file("second.csv") == csv_data
    # one chunk of the file maps to exactly one physical location
    first, second = store.info("first.csv"), store.info("second.csv")
    assert first.chunks == second.chunks
    assert first.hash == second.hash


def test_deduplication_of_modified_files(store, csv_data):
    store.write_file("v1.csv", csv_data)
    stored = store.index.total_stored_bytes()
    modified = csv_data + b"\n99999,appended,1\n" * 500
    store.write_file("v2.csv", modified)
    assert store.last_write["new_bytes"] < len(modified) - len(csv_data) + 200_000
    assert store.index.total_stored_bytes() - stored < len(modified)
    assert store.read_file("v2.csv") == modified
    stats = store.stats()
    assert stats.files == 2
    assert stats.dedup_bytes > 0
    assert 0 < stats.dedup_ratio < 1


def test_same_chunk_written_twice_in_one_file_is_stored_once(store):
    block = bytes(range(256)) * 200  # 51 KB, well above the minimum chunk size
    data = block * 20
    store.write_file("dup.bin", data)
    record = store.info("dup.bin")
    assert len({hash_hex for hash_hex, _ in record.chunks}) < record.nchunks
    assert store.read_file("dup.bin") == data


def test_index_is_reloadable(store, store_path, csv_data, binary_data):
    store.write_file("a.csv", csv_data)
    store.write_file("b.bin", binary_data)
    reopened = NXetStore.open(f"file://{store_path}", create=False)
    assert reopened.list_files() == ["a.csv", "b.bin"]
    assert reopened.read_file("a.csv") == csv_data
    assert reopened.read_file("b.bin") == binary_data
    assert reopened.stats() == store.stats()
    reopened.close()


def test_index_files_are_readable_json(store, csv_data):
    store.write_file("a.csv", csv_data)
    header = json.loads(open(store.index.path(HEADER_NAME), "rb").read())
    assert header["format"] == "nano-xet"
    assert header["chunking"]["mean"] == 65536
    assert header["chunking"]["min"] == 8192
    assert header["chunking"]["max"] == 131072

    xorb_lines = [
        json.loads(line)
        for line in open(store.index.path(XORBS_INDEX_NAME), "rb").read().split(b"\n")
        if line
    ]
    file_lines = [
        json.loads(line)
        for line in open(store.index.path(FILES_INDEX_NAME), "rb").read().split(b"\n")
        if line
    ]
    assert xorb_lines and xorb_lines[0]["xorb"].endswith(".xorb")
    assert file_lines[-1]["op"] == "put"
    assert file_lines[-1]["path"] == "a.csv"
    # the file -> xorb -> chunk links are consistent
    locations = {(h, o) for line in xorb_lines for h, _, o in line["chunks"]}
    assert len(locations) == sum(len(line["chunks"]) for line in xorb_lines)


def test_delete_and_gc(store, store_path, csv_data, binary_data):
    store.write_file("keep.csv", csv_data)
    store.write_file("drop.bin", binary_data)
    used_before = set(store.referenced_xorbs())
    store.delete_file("drop.bin")
    assert store.list_files() == ["keep.csv"]
    assert store.read_file("keep.csv") == csv_data
    removed = store.gc()
    assert removed and set(removed) == used_before - set(store.referenced_xorbs())
    assert store.read_file("keep.csv") == csv_data
    assert store.stats().files == 1
    # the store stays loadable after a gc
    store.reload()
    assert store.list_files() == ["keep.csv"]
    assert store.read_file("keep.csv") == csv_data


def test_gc_keeps_shared_xorbs(store, csv_data):
    store.write_file("a", csv_data)
    store.write_file("b", csv_data)  # fully deduplicated against a
    store.delete_file("a")
    assert store.gc(dry_run=True) == []  # b still references everything
    assert store.read_file("b") == csv_data


def test_unknown_file(store):
    with pytest.raises(FileNotFoundError):
        store.read_file("nope.csv")
    with pytest.raises(FileNotFoundError):
        store.info("nope.csv")
    with pytest.raises(FileNotFoundError):
        store.delete_file("nope.csv")
    store.delete_file("nope.csv", missing_ok=True)


def test_invalid_and_reserved_paths(store, csv_data):
    with pytest.raises(ValueError):
        store.write_file("../escape", csv_data)
    with pytest.raises(ValueError):
        store.write_file(HEADER_NAME, csv_data)
    with pytest.raises(ValueError):
        store.write_file("whatever.xorb", csv_data)
    assert store.normalize("/a//b/./c/") == "a/b/c"


def test_xorb_split_when_too_big(store_path, csv_data):
    store = NXetStore.open(f"file://{store_path}", max_xorb_bytes=64 * 1024)
    data = csv_data * 4
    store.write_file("big.csv", data)
    assert len(store.index.xorbs) > 1
    assert store.read_file("big.csv") == data
    for xorb in store.index.xorbs.values():
        assert xorb.nbytes <= 64 * 1024 + MAX_XORB_BYTES  # a xorb is bounded
    store.close()


def test_store_on_another_underlying_filesystem(csv_data):
    import fsspec

    mem = fsspec.filesystem("memory")
    store = NXetStore(mem, "/in-memory-store")
    store.write_file("dir/a.csv", csv_data)
    store.write_file("dir/b.csv", csv_data)
    assert store.read_file("dir/a.csv") == csv_data
    assert store.index.total_stored_bytes() == len(store.read_file("dir/a.csv"))
    names = [n for n in mem.ls("/in-memory-store", detail=False)]
    assert any(n.endswith(".xorb") for n in names)
    assert HEADER_NAME in [n.rsplit("/", 1)[-1] for n in names]
    store.close()


def test_not_a_store_error(store_path):
    with pytest.raises(Exception, match="not a nano-xet store"):
        NXetStore.open(f"file://{store_path}/does-not-exist", create=False)
