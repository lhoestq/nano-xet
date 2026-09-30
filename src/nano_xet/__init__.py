"""nano-xet: a tiny Xet-like content defined chunking + deduplication filesystem.

Quick start::

    export FSSPEC_NXET_STORE_URI=/tmp/my-nxet-store   # optional, see below

    import fsspec
    with fsspec.open("nxet://data/train.csv", "wb") as f:   # needs the env var
        f.write(b"hello nano-xet\\n")

    # ... or name the store inline, on any filesystem fsspec knows:
    with fsspec.open("nxet://data/train.csv::file:///tmp/my-nxet-store", "rb") as f:
        print(f.read())

See :class:`nano_xet.store.NXetStore` for the storage engine itself and
:class:`nano_xet.NXetFileSystem` for the ``nxet://`` fsspec filesystem.
"""

from __future__ import annotations

from .chunking import (
    GEAR_TABLE,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    Chunker,
    boundary_mask,
    chunk_iter,
    chunk_sizes,
)
from .fsspec_impl import NXetFile, NXetFileSystem
from .hashing import chunk_hash, merkle_hash, xorb_hash
from .index import FileRecord, NXetIndex, XorbRecord
from .store import NXetStore, Stats

__version__ = "0.1.0"

__all__ = [
    "Chunker",
    "FileRecord",
    "GEAR_TABLE",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "NXetFile",
    "NXetFileSystem",
    "NXetIndex",
    "NXetStore",
    "Stats",
    "TARGET_CHUNK_SIZE",
    "XorbRecord",
    "boundary_mask",
    "chunk_hash",
    "chunk_iter",
    "chunk_sizes",
    "merkle_hash",
    "xorb_hash",
]
