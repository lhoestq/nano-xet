"""The nano-xet index: the link files stored at the root of the underlying filesystem.

Layout of a nano-xet store (all names relative to the store root)::

    nxet.json           header: format version, chunking parameters, counters
    nxet.xorbs.jsonl    one line per xorb: xorb -> [(chunk hash, size, offset)]
    nxet.files.jsonl    append-only log: file -> [(chunk hash, size)], plus "rm" tombstones
    000000-<hex>.xorb   the xorb objects themselves (concatenated chunks)

Everything is JSON/JSONL so a demo store can be read and grepped by eye.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .chunking import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, TARGET_CHUNK_SIZE
from .hashing import HASH_ALGORITHM

HEADER_NAME = "nxet.json"
XORBS_INDEX_NAME = "nxet.xorbs.jsonl"
FILES_INDEX_NAME = "nxet.files.jsonl"
XORB_SUFFIX = ".xorb"
FORMAT_NAME = "nano-xet"
FORMAT_VERSION = 1


class NXetError(Exception):
    pass


class CorruptStoreError(NXetError):
    pass


@dataclass
class ChunkLocation:
    """Where a chunk lives physically: an offset inside a xorb."""

    xorb: str
    offset: int
    size: int


@dataclass
class XorbRecord:
    """A xorb: a physical object holding many chunks, ordered by chunk hash."""

    name: str
    size: int
    # [(chunk hash hex, chunk size, offset inside the xorb)] sorted by hash
    chunks: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def nbytes(self) -> int:
        return sum(size for _, size, _ in self.chunks)

    def to_json(self) -> dict:
        return {"xorb": self.name, "size": self.size, "chunks": self.chunks}

    @classmethod
    def from_json(cls, obj: dict) -> "XorbRecord":
        return cls(
            name=obj["xorb"],
            size=obj["size"],
            chunks=[(h, int(s), int(o)) for h, s, o in obj["chunks"]],
        )


@dataclass
class FileRecord:
    """A logical file: the ordered list of chunk hashes it is made of."""

    path: str
    size: int
    mtime: float
    hash: str
    chunks: List[Tuple[str, int]]  # [(chunk hash hex, chunk size)] in file order

    @property
    def nchunks(self) -> int:
        return len(self.chunks)

    def to_json(self, op: str = "put") -> dict:
        if op == "rm":
            return {"op": "rm", "path": self.path, "mtime": self.mtime}
        return {
            "op": op,
            "path": self.path,
            "size": self.size,
            "mtime": self.mtime,
            "hash": self.hash,
            "chunks": self.chunks,
        }

    @classmethod
    def from_json(cls, obj: dict) -> "FileRecord":
        return cls(
            path=obj["path"],
            size=int(obj.get("size", 0)),
            mtime=float(obj.get("mtime", 0.0)),
            hash=obj.get("hash", ""),
            chunks=[(h, int(s)) for h, s in obj.get("chunks", [])],
        )


class NXetIndex:
    """In-memory view of the store index, backed by the JSON link files."""

    def __init__(self, fs, root: str = "", header: Optional[dict] = None):
        self.fs = fs
        self.root = root.rstrip("/")
        self.header: dict = header or default_header()
        self.files: Dict[str, FileRecord] = {}
        self.xorbs: Dict[str, XorbRecord] = {}
        # chunk hash hex -> (xorb name, offset, size); a chunk is immutable, so
        # this map is append only.
        self.chunks: Dict[str, ChunkLocation] = {}
        # sizes of the index files when they were read: lets us notice that
        # another writer appended to them (None: never measured, assume stale)
        self._loaded_sizes: Optional[tuple] = None

    # -- paths ------------------------------------------------------------

    def path(self, name: str) -> str:
        return f"{self.root}/{name}" if self.root else name

    def xorb_path(self, xorb_name: str) -> str:
        return self.path(xorb_name)

    # -- loading ----------------------------------------------------------

    @classmethod
    def load(cls, fs, root: str = "", create: bool = False) -> "NXetIndex":
        index = cls(fs, root)
        try:
            raw = fs.cat_file(index.path(HEADER_NAME))
        except FileNotFoundError:
            if not create:
                raise NXetError(
                    f"{root or '.'} is not a nano-xet store ({HEADER_NAME} not found)"
                )
            index.save_header()
            index._append(FILES_INDEX_NAME, "")  # touch the two index files
            index._append(XORBS_INDEX_NAME, "")
            index._loaded_sizes = index._index_sizes()
            return index
        header = json.loads(raw or b"{}")
        if header.get("format") != FORMAT_NAME:
            raise CorruptStoreError(f"{root or '.'}: not a nano-xet store (header={header})")
        if header.get("version", 0) > FORMAT_VERSION:
            raise CorruptStoreError(
                f"store {root or '.'} was written by a newer nano-xet "
                f"(version {header['version']} > {FORMAT_VERSION})"
            )
        index.header = header
        index._load_xorbs()
        index._load_files()
        index._loaded_sizes = index._index_sizes()
        return index

    def _index_sizes(self) -> tuple:
        """Sizes of the two index files, used to notice writes of another writer."""
        sizes = []
        for name in (FILES_INDEX_NAME, XORBS_INDEX_NAME):
            try:
                sizes.append(self.fs.info(self.path(name))["size"])
            except (FileNotFoundError, OSError):
                sizes.append(None)
        return tuple(sizes)

    def is_stale(self) -> bool:
        """True when the index files changed on the underlying filesystem."""
        if self._loaded_sizes is None:
            return True
        return self._loaded_sizes != self._index_sizes()

    def _load_xorbs(self) -> None:
        for obj in self._read_lines(XORBS_INDEX_NAME):
            record = XorbRecord.from_json(obj)
            self.xorbs[record.name] = record
            for h, size, offset in record.chunks:
                self.chunks[h] = ChunkLocation(record.name, offset, size)

    def _load_files(self) -> None:
        for obj in self._read_lines(FILES_INDEX_NAME):
            record = FileRecord.from_json(obj)
            if obj.get("op") == "rm":
                self.files.pop(record.path, None)
            else:
                self.files[record.path] = record

    # -- persistence ------------------------------------------------------

    def _append(self, name: str, line: str) -> None:
        """Append a line of JSON, with a fallback for filesystems without append."""
        target = self.path(name)
        payload = line.encode() if isinstance(line, str) else line
        try:
            with self.fs.open(target, "ab") as f:
                f.write(payload)
        except (NotImplementedError, FileNotFoundError, ValueError):
            # memory:// and most object stores cannot append: read + rewrite instead
            try:
                existing = self.fs.cat_file(target)
            except FileNotFoundError:
                existing = b""
            self.fs.pipe_file(target, existing + payload)
            self._remember_size(name, len(existing) + len(payload))
            return
        self._remember_size(name, None if self._loaded_sizes is None else -1)

    def _remember_size(self, name: str, size: Optional[int]) -> None:
        """Keep the recorded index sizes in sync with our own appends."""
        if self._loaded_sizes is None:
            return
        index = 0 if name == FILES_INDEX_NAME else 1
        sizes = list(self._loaded_sizes)
        current = sizes[index] or 0
        sizes[index] = current + 1 if size is None or size < 0 else size
        self._loaded_sizes = tuple(sizes)

    @staticmethod
    def _json_line(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":")) + "\n"

    def _read_lines(self, name: str):
        try:
            raw = self.fs.cat_file(self.path(name))
        except FileNotFoundError:
            return
        for line in raw.split(b"\n"):
            if line.strip():
                yield json.loads(line)

    def save_header(self) -> None:
        with self.fs.open(self.path(HEADER_NAME), "wb") as f:
            f.write(json.dumps(self.header, indent=2).encode() + b"\n")

    def commit_xorb(self, record: XorbRecord) -> None:
        """Append a xorb to the index (the object must already be written)."""
        self.xorbs[record.name] = record
        for h, size, offset in record.chunks:
            self.chunks[h] = ChunkLocation(record.name, offset, size)
        self._append(XORBS_INDEX_NAME, self._json_line(record.to_json()))
        self.header["next_xorb_seq"] = self.header.get("next_xorb_seq", 0) + 1
        self.save_header()

    def commit_file(self, record: FileRecord) -> None:
        self.files[record.path] = record
        self._append(FILES_INDEX_NAME, self._json_line(record.to_json(op="put")))

    def log_remove(self, path: str) -> None:
        self.files.pop(path, None)
        self._append(
            FILES_INDEX_NAME,
            self._json_line({"op": "rm", "path": path, "mtime": time.time()}),
        )

    # -- helpers ----------------------------------------------------------

    def next_xorb_name(self, hash_hex: str) -> str:
        seq = self.header.get("next_xorb_seq", 0)
        return f"{seq:06d}-{hash_hex[:16]}{XORB_SUFFIX}"

    @property
    def chunking(self) -> dict:
        return self.header.get("chunking", {})

    def total_stored_bytes(self) -> int:
        return sum(x.nbytes for x in self.xorbs.values())


def default_header() -> dict:
    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "hash": HASH_ALGORITHM,
        "chunking": {
            "mean": TARGET_CHUNK_SIZE,
            "min": MIN_CHUNK_SIZE,
            "max": MAX_CHUNK_SIZE,
            "algorithm": "gearhash (same table and boundary mask as Xet)",
        },
        "next_xorb_seq": 0,
    }
