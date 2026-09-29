"""The nano-xet store: chunking + deduplication + xorbs on an underlying filesystem.

Write path (same shape as Xet's)::

    data -> gear-hash chunks -> chunk hash -> dedup lookup
                                            |-> known hash: reuse, nothing stored
                                            `-> new hash:  buffered in the current xorb
    full xorb -> sorted by chunk hash -> written as one physical object
    file -> [(chunk hash, size), ...] -> appended to the files index

Read path::

    file -> [(chunk hash, size), ...] -> chunk index -> (xorb, offset, size)
       -> coalesced reads of neighbouring chunks in the same xorb
"""

from __future__ import annotations

import posixpath
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from .chunking import chunk_iter
from .hashing import chunk_hash, merkle_hash, xorb_hash
from .index import (
    FILES_INDEX_NAME,
    HEADER_NAME,
    XORBS_INDEX_NAME,
    FileRecord,
    NXetIndex,
    XorbRecord,
)

# xet_core_structures/src/xorb_object/constants.rs
MAX_XORB_BYTES = 64 * 1024 * 1024
MAX_XORB_CHUNKS = 8 * 1024

RESERVED_NAMES = (HEADER_NAME, FILES_INDEX_NAME, XORBS_INDEX_NAME)


@dataclass
class Stats:
    """Deduplication statistics of a store."""

    files: int = 0
    logical_bytes: int = 0
    chunks: int = 0
    dedup_chunks: int = 0
    dedup_bytes: int = 0
    stored_chunks: int = 0
    stored_bytes: int = 0
    xorbs: int = 0

    @property
    def dedup_ratio(self) -> float:
        """Fraction of the logical bytes that did not have to be stored."""
        if self.logical_bytes == 0:
            return 0.0
        return self.dedup_bytes / self.logical_bytes

    @property
    def stored_over_logical(self) -> float:
        if self.logical_bytes == 0:
            return 0.0
        return self.stored_bytes / self.logical_bytes

    def to_dict(self) -> dict:
        return {
            "files": self.files,
            "logical_bytes": self.logical_bytes,
            "chunks": self.chunks,
            "dedup_chunks": self.dedup_chunks,
            "dedup_bytes": self.dedup_bytes,
            "stored_chunks": self.stored_chunks,
            "stored_bytes": self.stored_bytes,
            "xorbs": self.xorbs,
            "dedup_ratio": self.dedup_ratio,
        }

    def summary(self) -> str:
        saved = 100.0 * self.dedup_ratio
        return "\n".join(
            [
                f"{self.files} file(s), {self.chunks} chunk(s), {self.xorbs} xorb(s)",
                f"logical : {human_bytes(self.logical_bytes)}",
                f"stored  : {human_bytes(self.stored_bytes)}"
                f" ({self.stored_chunks} unique chunk(s))",
                f"dedup   : {human_bytes(self.dedup_bytes)} saved ({saved:.1f}%)"
                f" [{self.dedup_chunks} chunk(s) reused]",
            ]
        )


def human_bytes(size: float) -> str:
    """``2149408`` -> ``2.1 MB`` (decimal units, like storage vendors)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


class XorbBuilder:
    """Buffers new (not yet stored) chunks until the xorb is full."""

    def __init__(
        self,
        max_bytes: int = MAX_XORB_BYTES,
        max_chunks: int = MAX_XORB_CHUNKS,
    ):
        self.max_bytes = max_bytes
        self.max_chunks = max_chunks
        self._items: List[Tuple[str, bytes]] = []
        self._seen: set[str] = set()
        self.nbytes = 0

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def contains(self, hash_hex: str) -> bool:
        return hash_hex in self._seen

    def add(self, hash_hex: str, data: bytes) -> None:
        if hash_hex in self._seen:
            return
        self._seen.add(hash_hex)
        self._items.append((hash_hex, data))
        self.nbytes += len(data)

    @property
    def full(self) -> bool:
        return self.nbytes >= self.max_bytes or len(self._items) >= self.max_chunks

    def build(self) -> Tuple[bytes, XorbRecord]:
        """Concatenate the chunks ordered by hash (as Xet does inside a xorb)."""
        items = sorted(self._items, key=lambda item: item[0])
        payload = bytearray()
        chunks: List[Tuple[str, int, int]] = []
        for hash_hex, data in items:
            chunks.append((hash_hex, len(data), len(payload)))
            payload += data
        return bytes(payload), chunks


class NXetStore:
    """A nano-xet store living at ``root`` of the ``fs`` underlying filesystem."""

    def __init__(
        self,
        fs,
        root: str = "",
        create: bool = True,
        max_xorb_bytes: int = MAX_XORB_BYTES,
        max_xorb_chunks: int = MAX_XORB_CHUNKS,
        use_numpy: bool = True,
        max_open_xorbs: int = 16,
    ):
        self.fs = fs
        self.root = root.rstrip("/")
        self.max_xorb_bytes = max_xorb_bytes
        self.max_xorb_chunks = max_xorb_chunks
        self.use_numpy = use_numpy
        self.max_open_xorbs = max_open_xorbs
        if create and self.root:
            try:
                fs.makedirs(self.root, exist_ok=True)
            except (OSError, NotImplementedError):
                pass  # some filesystems have no real directories
        self.index = NXetIndex.load(fs, self.root, create=create)
        # counters, reset for every write operation
        self.last_write: Dict[str, int] = {}
        self._handles: Dict[str, object] = {}

    # -- construction -----------------------------------------------------

    @classmethod
    def open(cls, uri: str, create: bool = True, **kwargs) -> "NXetStore":
        """Open ``nxet://<path>::<underlying uri>`` (or just ``<underlying uri>``)."""
        import fsspec

        underlying = uri.split("::", 1)[1] if "::" in uri else uri
        if not underlying:
            raise ValueError(
                "a nano-xet store needs an underlying filesystem, e.g. "
                "'nxet://data.csv::file:///tmp/my-nxet-store'"
            )
        fs, root = fsspec.core.url_to_fs(underlying)
        return cls(fs, root, create=create, **kwargs)

    # -- paths ------------------------------------------------------------

    @staticmethod
    def normalize(path: str) -> str:
        parts = [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]
        if ".." in parts:
            raise ValueError(f"invalid nano-xet path: {path!r}")
        return posixpath.join(*parts) if parts else ""

    def _check_writable(self, path: str) -> None:
        if path in RESERVED_NAMES or path.endswith(".xorb"):
            raise ValueError(f"{path!r} collides with a nano-xet index file")

    # -- writing ----------------------------------------------------------

    def write_file(self, path: str, data: bytes) -> FileRecord:
        """Chunk ``data``, store the new chunks and link the file to them."""
        path = self.normalize(path)
        if not path:
            raise ValueError("cannot write the store root")
        self._check_writable(path)
        if isinstance(data, (bytearray, memoryview)):
            data = bytes(data)

        pending = XorbBuilder(self.max_xorb_bytes, self.max_xorb_chunks)
        file_chunks: List[Tuple[str, int]] = []
        hashes: List[bytes] = []
        dedup_chunks = dedup_bytes = 0
        new_chunks = new_bytes = 0

        for chunk in chunk_iter(data, use_numpy=self.use_numpy):
            hash_hex = chunk_hash(chunk).hex()
            file_chunks.append((hash_hex, len(chunk)))
            hashes.append(bytes.fromhex(hash_hex))
            if hash_hex in self.index.chunks or pending.contains(hash_hex):
                dedup_chunks += 1
                dedup_bytes += len(chunk)
                continue
            pending.add(hash_hex, chunk)
            new_chunks += 1
            new_bytes += len(chunk)
            if pending.full:
                self._flush_xorb(pending)
                pending = XorbBuilder(self.max_xorb_bytes, self.max_xorb_chunks)
        if pending:
            self._flush_xorb(pending)

        record = FileRecord(
            path=path,
            size=len(data),
            mtime=time.time(),
            hash=merkle_hash(hashes).hex(),
            chunks=file_chunks,
        )
        self.index.commit_file(record)
        self.last_write = {
            "size": len(data),
            "chunks": len(file_chunks),
            "dedup_chunks": dedup_chunks,
            "dedup_bytes": dedup_bytes,
            "new_chunks": new_chunks,
            "new_bytes": new_bytes,
            "file_hash": record.hash,
        }
        return record

    def _flush_xorb(self, builder: XorbBuilder) -> XorbRecord:
        payload, chunks = builder.build()
        name = self.index.next_xorb_name(xorb_hash([bytes.fromhex(h) for h, _, _ in chunks]).hex())
        target = self.index.xorb_path(name)
        if self.root:
            self.fs.makedirs(self.root, exist_ok=True)
        with self.fs.open(target, "wb") as f:
            f.write(payload)
        record = XorbRecord(name=name, size=len(payload), chunks=chunks)
        self.index.commit_xorb(record)
        return record

    # -- reading ----------------------------------------------------------

    def info(self, path: str) -> FileRecord:
        record = self.index.files.get(self.normalize(path))
        if record is None:
            raise FileNotFoundError(path)
        return record

    def list_files(self) -> List[str]:
        return sorted(self.index.files)

    def read_file(self, path: str) -> bytes:
        return self.read_range(path, 0, None)

    def read_range(self, path: str, start: int, end: Optional[int] = None) -> bytes:
        record = self.info(path)
        size = record.size
        start = max(0, int(start))
        end = size if end is None else min(int(end), size)
        if start >= end:
            return b""
        pieces = [
            self._read_xorb_range(xorb, offset, nbytes)
            for xorb, offset, nbytes in self._coalesced_reads(record, start, end)
        ]
        return b"".join(pieces)

    def _coalesced_reads(
        self, record: FileRecord, start: int, end: int
    ) -> Iterator[Tuple[str, int, int]]:
        """Physical ``(xorb, offset, size)`` reads covering ``[start, end)``, in file order.

        Neighbouring chunks that happen to be stored next to each other in the same
        xorb are read in a single pass.
        """
        runs: List[List[int | str]] = []
        pos = 0
        for hash_hex, chunk_size in record.chunks:
            nxt = pos + chunk_size
            if nxt > start and pos < end:
                loc = self.index.chunks.get(hash_hex)
                if loc is None:
                    raise KeyError(
                        f"chunk {hash_hex} referenced by {record.path} is missing from the store"
                    )
                offset = loc.offset + max(0, start - pos)
                nbytes = min(chunk_size, end - pos) - max(0, start - pos)
                if runs and runs[-1][0] == loc.xorb and runs[-1][1] + runs[-1][2] == offset:
                    runs[-1][2] += nbytes
                else:
                    runs.append([loc.xorb, offset, nbytes])
            pos = nxt
        for xorb, offset, nbytes in runs:
            yield str(xorb), int(offset), int(nbytes)

    def _read_xorb_range(self, name: str, offset: int, size: int) -> bytes:
        handle = self._handles.get(name)
        if handle is None:
            if len(self._handles) >= self.max_open_xorbs:
                for old in list(self._handles)[: len(self._handles) - self.max_open_xorbs + 1]:
                    self.close_handle(old)
            handle = self.fs.open(self.index.xorb_path(name), "rb")
            self._handles[name] = handle
        handle.seek(offset)
        data = handle.read(size)
        if len(data) != size:
            raise IOError(f"short read of xorb {name}: {len(data)} != {size}")
        return data

    def close_handle(self, name: str) -> None:
        handle = self._handles.pop(name, None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def close(self) -> None:
        for name in list(self._handles):
            self.close_handle(name)

    def reload(self) -> None:
        """Re-read the index files (another writer may have updated the store)."""
        self.close()
        self.index = NXetIndex.load(self.fs, self.root, create=False)

    def reload_if_stale(self) -> bool:
        """Reload when the index files changed since they were read. True if reloaded."""
        if self.index.is_stale():
            self.reload()
            return True
        return False

    # -- deleting / maintenance --------------------------------------------

    def delete_file(self, path: str, missing_ok: bool = False) -> None:
        path = self.normalize(path)
        if path not in self.index.files:
            if missing_ok:
                return
            raise FileNotFoundError(path)
        self.index.log_remove(path)

    def referenced_xorbs(self) -> set:
        used = set()
        for record in self.index.files.values():
            for hash_hex, _ in record.chunks:
                loc = self.index.chunks.get(hash_hex)
                if loc is not None:
                    used.add(loc.xorb)
        return used

    def gc(self, dry_run: bool = False) -> List[str]:
        """Delete the xorbs no longer referenced by any file (after ``delete_file``).

        The two index files are rewritten from the in-memory index.
        """
        used = self.referenced_xorbs()
        doomed = sorted(set(self.index.xorbs) - used)
        if dry_run or not doomed:
            return doomed
        for name in doomed:
            self.fs.rm(self.index.xorb_path(name))
            self.close_handle(name)
            self.index.xorbs.pop(name, None)
        for hash_hex, loc in list(self.index.chunks.items()):
            if loc.xorb not in self.index.xorbs:
                self.index.chunks.pop(hash_hex, None)
        self.index.header["next_xorb_seq"] = len(self.index.xorbs)
        self._rewrite_indexes()
        return doomed

    def _rewrite_indexes(self) -> None:
        xorbs = "".join(
            self.index._json_line(r.to_json())
            for r in sorted(self.index.xorbs.values(), key=lambda r: r.name)
        )
        files = "".join(
            self.index._json_line(r.to_json())
            for r in sorted(self.index.files.values(), key=lambda r: r.path)
        )
        with self.fs.open(self.index.path(XORBS_INDEX_NAME), "wb") as f:
            f.write(xorbs.encode())
        with self.fs.open(self.index.path(FILES_INDEX_NAME), "wb") as f:
            f.write(files.encode())
        self.index.save_header()

    # -- stats ------------------------------------------------------------

    def header(self) -> dict:
        """Format, hash algorithm and chunking parameters of the store."""
        return dict(self.index.header)

    def stats(self) -> Stats:
        stats = Stats()
        stats.files = len(self.index.files)
        stats.xorbs = len(self.index.xorbs)
        stats.stored_chunks = len(self.index.chunks)
        stats.stored_bytes = self.index.total_stored_bytes()
        for record in self.index.files.values():
            stats.logical_bytes += record.size
            stats.chunks += record.nchunks
        stats.dedup_chunks = stats.chunks - stats.stored_chunks
        stats.dedup_bytes = stats.logical_bytes - stats.stored_bytes
        return stats

    def __repr__(self) -> str:
        protocol = self.fs.protocol
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0]
        return f"NXetStore({protocol}://{self.root})"

    def __enter__(self) -> "NXetStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
