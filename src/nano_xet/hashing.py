"""Hashing helpers.

Xet identifies every chunk with a keyed ``blake3`` hash and every file with a
merkle hash over its chunk hashes. nano-xet keeps the same *roles* (chunk hash,
xorb hash, file hash) but uses ``blake2b-256`` from the standard library
(``hashlib``) instead of keyed blake3, so that it stays dependency free.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

HASH_ALGORITHM = "blake2b-256"
HASH_SIZE = 32


def digest(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=HASH_SIZE).digest()


def chunk_hash(data: bytes) -> bytes:
    """Hash of a single chunk - the deduplication key (Xet: keyed blake3)."""
    return digest(data)


def merkle_hash(chunk_hashes: Iterable[bytes]) -> bytes:
    """Hash of a sequence of chunk hashes, standing in for the Xet merkle hash."""
    h = hashlib.blake2b(digest_size=HASH_SIZE)
    count = 0
    for chunk in chunk_hashes:
        h.update(chunk)
        count += 1
    h.update(count.to_bytes(8, "little"))
    return h.digest()


def xorb_hash(chunk_hashes: Iterable[bytes]) -> bytes:
    """Content hash of a xorb: the hash of the list of its chunk hashes."""
    return merkle_hash(chunk_hashes)
