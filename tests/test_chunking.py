"""Chunking must produce exactly the chunks Xet would produce.

The golden values below were produced by the reference rust chunker of
``xet-core`` (``xet_data/src/deduplication/chunking.rs`` + the ``gearhash``
crate) on the same deterministic inputs.
"""

import random

import pytest

from nano_xet.chunking import (
    GEAR_TABLE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    Chunker,
    boundary_mask,
    chunk_iter,
    chunk_sizes,
)


def _random_data(size=300_000, seed=42):
    return random.Random(seed).randbytes(size)


def _text_data(rows=6000, seed=42):
    rng = random.Random(seed)
    return "".join(f"{i},{i * i % 9973},{rng.random():.6f}\n" for i in range(rows)).encode()


def test_gear_table_is_the_xet_one():
    assert len(GEAR_TABLE) == 256
    # first, middle and last entries of gearhash::DEFAULT_TABLE
    assert GEAR_TABLE[0] == 0xB088D3A9E840F559
    assert GEAR_TABLE[128] == 0xA315F5EBFB706D26
    assert GEAR_TABLE[-1] == 0x63C7A906C1DD187B
    assert all(0 <= value < 2**64 for value in GEAR_TABLE)


def test_xet_chunk_sizes():
    assert TARGET_CHUNK_SIZE == 64 * 1024
    assert MIN_CHUNK_SIZE == 8 * 1024  # mean / MINIMUM_CHUNK_DIVISOR
    assert MAX_CHUNK_SIZE == 128 * 1024  # mean * MAXIMUM_CHUNK_MULTIPLIER


def test_boundary_mask_is_shifted_to_the_top_bits():
    assert boundary_mask() == 0xFFFF000000000000
    assert boundary_mask(4096) == 0xFFF0000000000000


@pytest.mark.parametrize("invalid", [3000, 0, 64, 32, -8])
def test_boundary_mask_rejects_invalid_targets(invalid):
    with pytest.raises(ValueError):
        boundary_mask(invalid)


@pytest.mark.parametrize(
    "data, expected",
    [
        (_random_data(), [54905, 131072, 98479, 15544]),
        (_text_data(), [49874, 17654, 44225, 475]),
    ],
    ids=["random-300k", "csv-6000rows"],
)
def test_chunk_sizes_match_xet(data, expected):
    assert sum(chunk_sizes(data)) == len(data)
    assert chunk_sizes(data, use_numpy=False) == expected
    assert chunk_sizes(data, use_numpy=True) == expected


@pytest.mark.parametrize("size", [0, 1, 100, 8191, 8192, 8193, 65536, 131072, 131073, 250_000])
def test_chunk_sizes_against_reference_for_edge_sizes(size):
    # reference boundaries, same rust chunker as above
    reference = _reference_sizes(size)
    data = _random_data(size, seed=size)
    assert chunk_sizes(data, use_numpy=False) == reference
    assert chunk_sizes(data, use_numpy=True) == reference


def _reference_sizes(size):
    """Chunk sizes recorded from the rust chunker for seed=size, len=size."""
    return _REFERENCE.get(size)


_REFERENCE = {
    0: [],
    1: [1],
    100: [100],
    8191: [8191],
    8192: [8192],
    8193: [8193],
    65536: [65536],
    131072: [99306, 31766],
    131073: [55996, 60013, 15064],
    250_000: [39930, 10143, 88601, 45528, 47636, 11220, 6942],
}


def test_chunks_respect_min_and_max():
    data = _random_data(1_000_000, seed=7)
    sizes = chunk_sizes(data)
    assert sum(sizes) == len(data)
    assert all(MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE for size in sizes[:-1])
    assert sizes[-1] <= MAX_CHUNK_SIZE
    # a mean of ~64 KB is the point of the mask
    assert 55_000 < sum(sizes) / len(sizes) < 75_000


def test_streaming_chunking_matches_whole_buffer():
    data = _random_data(900_000, seed=99)
    expected = chunk_sizes(data, use_numpy=False)
    chunker = Chunker(use_numpy=False)
    got = []
    for start in range(0, len(data), 37_000):
        got += [len(c) for c in chunker.next_block(data[start : start + 37_000])]
    got += [len(c) for c in chunker.next_block(b"", is_final=True)]
    assert got == expected


def test_numpy_path_matches_the_scalar_one():
    for size in (100_000, 300_000, 1_200_000):
        data = _random_data(size, seed=size + 1)
        assert chunk_sizes(data, use_numpy=False) == chunk_sizes(data, use_numpy=True)


def test_chunk_boundaries_do_not_move_when_appending():
    """Appending data never moves the boundaries found before the new data."""
    data = _random_data(400_000, seed=5)
    extended = data + _random_data(200_000, seed=6)
    assert chunk_sizes(data)[:-1] == chunk_sizes(extended)[: len(chunk_sizes(data)) - 1]


def test_local_edit_keeps_most_chunk_hashes():
    """A change in the middle only invalidates the chunks around it."""
    from nano_xet.hashing import chunk_hash

    data = _random_data(800_000, seed=11)
    edited = data[:400_000] + b"nano-xet!" * 500 + data[400_000:]
    before = {chunk_hash(c) for c in chunk_iter(data)}
    after = {chunk_hash(c) for c in chunk_iter(edited)}
    assert len(before & after) > 0.8 * len(before)


def test_empty_input():
    assert list(chunk_iter(b"")) == []
    assert chunk_sizes(b"") == []
