"""fsspec filesystem exposing a nano-xet store with the ``nxet://`` protocol.

The URI is a *chained* URI (like the zip/tar ones), the second part being the
underlying filesystem the xorbs are stored on::

    nxet://path/inside/the/store/data.csv::file:///tmp/my-nxet-store
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^ the "underlying filesystem"

Any filesystem fsspec can open works as the underlying filesystem::

    nxet://train.parquet::s3://my-bucket/nxet-store
    nxet://train.parquet::memory://nxet-store
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import fsspec
from fsspec.spec import AbstractBufferedFile

from .store import MAX_XORB_BYTES, MAX_XORB_CHUNKS, NXetStore, Stats

PROTOCOL = "nxet"


def split_chain(uri: str) -> tuple:
    """``"a/b::file:///tmp/s"`` -> ``("a/b", "file:///tmp/s")``."""
    if "::" in uri:
        inner, _, outer = uri.partition("::")
        return inner, outer
    return uri, ""


def underlying_location(fo: str) -> str:
    """Where the xorbs live, for any spelling of ``fo``.

    ``"/tmp/store"``, ``"file:///tmp/store"`` and
    ``"nxet://a/b.csv::file:///tmp/store"`` all point at ``/tmp/store``; a bare
    ``nxet://a/b.csv`` does not say where the store is, and returns "".
    """
    path, extra = split_chain(str(fo))
    if extra:
        return extra.split("::")[-1]
    return "" if path.startswith(f"{PROTOCOL}:") else path


def store_location(fo: str = "", store_uri: str = "") -> str:
    """Store location from the arguments of ``NXetFileSystem``.

    An explicit ``fo`` (or a chained ``nxet://`` uri) wins over ``store_uri``,
    the fallback meant for ``FSSPEC_NXET_STORE_URI``.
    """
    return underlying_location(fo or "") or underlying_location(store_uri or "")


class NXetFileSystem(fsspec.AbstractFileSystem):
    """Read/write files stored in nano-xet (gear-hash chunks + xorbs + dedup).

    Parameters
    ----------
    fo: str
        Path of the store inside the underlying filesystem: either
        ``"/tmp/my-store"`` together with ``target_protocol="file"``, or a full
        url such as ``"file:///tmp/my-store"``. This is what follows the ``::``
        of a chained ``nxet://`` URI.
    store_uri: str
        Alias of ``fo`` for the case where the uri only carries the file path
        (``nxet://data/train.csv``). fsspec fills keyword arguments from the
        environment, so ``FSSPEC_NXET_STORE_URI=/tmp/my-store`` is enough to
        work with plain ``nxet://`` paths; an explicit ``fo`` still wins.
    target_protocol: str
        Protocol of the underlying filesystem; inferred from ``fo`` if not given.
    target_options: dict
        Extra arguments for the underlying filesystem.
    create: bool
        Create the store (its index files) if it does not exist yet.
    use_numpy: bool
        Use the vectorised gear hash (numpy) when available: ~4x faster chunking.
    """

    protocol = PROTOCOL
    root_marker = ""
    cachable = True

    #: one instance per underlying store, whatever the spelling of ``fo`` was
    _instances: Dict[tuple, "NXetFileSystem"] = {}

    def __new__(cls, *args, **kwargs):
        key = cls._store_key(args, kwargs)
        known = cls._instances.get(key) if key is not None else None
        return known if known is not None else super().__new__(cls)

    @classmethod
    def _store_key(cls, args: tuple, kwargs: dict) -> Optional[tuple]:
        """Canonical (protocol, root) of the store an instance would open."""
        fo = kwargs.get("fo", args[0] if args else "")
        target_options = dict(kwargs.get("target_options") or {})
        target_protocol = kwargs.get("target_protocol")
        location = store_location(fo, kwargs.get("store_uri", ""))
        if not location and target_protocol is None:
            return None
        try:
            if target_protocol is None:
                underlying_fs, root = fsspec.core.url_to_fs(location, **target_options)
            else:
                underlying_fs = fsspec.filesystem(target_protocol, **target_options)
                root = underlying_fs._strip_protocol(location)
        except Exception:  # bad url: let __init__ produce the real error
            return None
        return _protocol_of(underlying_fs), root

    @classmethod
    def clear_instance_cache(cls) -> None:
        super().clear_instance_cache()
        cls._instances.clear()

    def __init__(
        self,
        fo="",
        target_protocol: Optional[str] = None,
        target_options: Optional[dict] = None,
        create: bool = True,
        use_numpy: bool = True,
        max_xorb_bytes: int = MAX_XORB_BYTES,
        max_xorb_chunks: int = MAX_XORB_CHUNKS,
        max_open_xorbs: int = 16,
        store_uri: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        location = store_location(fo, store_uri or "")
        target_options = dict(target_options or {})
        if not location and target_protocol is None:
            raise ValueError(
                "nxet:// needs an underlying filesystem to store its xorbs, e.g. "
                "'nxet://data.csv::file:///tmp/my-store', "
                "NXetFileSystem(store_uri='/tmp/my-store'), or the "
                "FSSPEC_NXET_STORE_URI environment variable"
            )
        if target_protocol is None:
            underlying_fs, root = fsspec.core.url_to_fs(location, **target_options)
        else:
            # an empty location is valid here: the store sits at the root of that fs
            underlying_fs = fsspec.filesystem(target_protocol, **target_options)
            root = underlying_fs._strip_protocol(location)
        if getattr(self, "store", None) is not None:
            self.store.close()  # re-initialising a known store: drop old handles
        self.underlying_fs = underlying_fs
        self.root = root
        self._uri_store = f"{_protocol_of(underlying_fs)}://{root}"
        self.store = NXetStore(
            underlying_fs,
            root,
            create=create,
            use_numpy=use_numpy,
            max_xorb_bytes=max_xorb_bytes,
            max_xorb_chunks=max_xorb_chunks,
            max_open_xorbs=max_open_xorbs,
        )
        self._instances[(self._uri_store.rsplit("://", 1)[0], root)] = self

    # -- URIs -------------------------------------------------------------

    @classmethod
    def _strip_protocol(cls, path) -> str:
        path = split_chain(str(path))[0]
        if path.startswith(f"{PROTOCOL}://"):
            path = path[len(f"{PROTOCOL}://") :]
        return path.lstrip("/").rstrip("/")

    def unstrip_protocol(self, path: str) -> str:
        """Full chained ``nxet://`` URI of a path of this store."""
        return f"{PROTOCOL}://{self._strip_protocol(path)}::{self._uri_store}"

    url = unstrip_protocol

    # -- metadata ---------------------------------------------------------

    @staticmethod
    def _dir_info(path: str) -> dict:
        return {"name": path, "size": None, "type": "directory"}

    @staticmethod
    def _file_info(record) -> dict:
        return {
            "name": record.path,
            "size": record.size,
            "type": "file",
            "mtime": record.mtime,
            "created": record.mtime,
            "nchunks": record.nchunks,
            "xhash": record.hash,
        }

    def _is_known_dir(self, path: str) -> bool:
        path = path.rstrip("/")
        prefix = path + "/"
        return any(name.startswith(prefix) for name in self.store.index.files)

    def _retry_when_missing(self, read, *args, **kwargs):
        """Retry a read once after reloading the index (written by another process)."""
        try:
            return read(*args, **kwargs)
        except FileNotFoundError:
            if self.store.reload_if_stale():
                return read(*args, **kwargs)
            raise

    def info(self, path, **kwargs) -> dict:
        return self._retry_when_missing(self._info, self._strip_protocol(path))

    def _info(self, path: str) -> dict:
        record = self.store.index.files.get(path)
        if record is not None:
            return self._file_info(record)
        if not path or self._is_known_dir(path):
            return self._dir_info(path)
        raise FileNotFoundError(path)

    def ls(self, path, detail: bool = True, **kwargs) -> List:
        return self._retry_when_missing(self._ls, self._strip_protocol(path), detail)

    def _ls(self, path: str, detail: bool = True) -> List:
        files = self.store.index.files
        if path and path in files:
            entries = [self._file_info(files[path])]
        elif not path or self._is_known_dir(path):
            prefix = path + "/" if path else ""
            found: Dict[str, dict] = {}
            for full in files:
                if not full.startswith(prefix):
                    continue
                rest = full[len(prefix) :]
                if "/" in rest:
                    name = prefix + rest.split("/")[0]
                    found.setdefault(name, self._dir_info(name))
                else:
                    found[full] = self._file_info(files[full])
            entries = [found[name] for name in sorted(found)]
            if not entries and path:
                raise FileNotFoundError(path)
        else:
            raise FileNotFoundError(path)
        return entries if detail else [entry["name"] for entry in entries]

    def isdir(self, path) -> bool:
        path = self._strip_protocol(path)
        return not path or self._is_known_dir(path)

    def created(self, path):
        return self.info(path)["created"]

    def modified(self, path):
        return self.info(path)["mtime"]

    # -- reading / writing ------------------------------------------------

    def cat_file(self, path, start=None, end=None, **kwargs) -> bytes:
        path = self._strip_protocol(path)
        return self._retry_when_missing(self.store.read_range, path, start or 0, end)

    def pipe_file(self, path, value, mode: str = "wb", **kwargs) -> None:
        path = self._strip_protocol(path)
        if mode == "ab":
            try:
                value = self.store.read_file(path) + value
            except FileNotFoundError:
                pass
        self.store.write_file(path, value)
        self.invalidate_cache(self._parent(path))

    def cp_file(self, path1, path2, **kwargs) -> None:
        path1 = self._strip_protocol(path1)
        if self.isdir(path1):
            return  # virtual directory: it exists as soon as its files are copied
        self.pipe_file(path2, self.cat_file(path1))

    def get_file(self, path1, path2, callback=None, **kwargs) -> None:
        """Copy a file of the store to the local filesystem."""
        path1 = self._strip_protocol(path1)
        if self.isdir(path1):
            os.makedirs(str(path2), exist_ok=True)  # the generic get() walks the tree
            return
        data = self.store.read_file(path1)
        parent = os.path.dirname(str(path2))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path2, "wb") as f:
            f.write(data)

    def put_file(self, path1, path2, callback=None, **kwargs) -> None:
        """Copy a local file into the store."""
        with open(path1, "rb") as f:
            self.pipe_file(path2, f.read())

    # -- deleting ---------------------------------------------------------

    def rm_file(self, path) -> None:
        path = self._strip_protocol(path)
        try:
            self.store.delete_file(path)
        except FileNotFoundError:
            if not self.store.reload_if_stale() or path not in self.store.index.files:
                raise
            self.store.delete_file(path)
        self.invalidate_cache(self._parent(path))

    def rm(self, path, recursive: bool = False, maxdepth=None) -> None:
        if isinstance(path, (list, tuple)):
            for one in path:
                self.rm(one, recursive=recursive)
            return
        path = self._strip_protocol(path)
        if path in self.store.index.files:
            self.rm_file(path)
            return
        children = [name for name in self.store.index.files if name.startswith(path + "/")]
        if not children:
            raise FileNotFoundError(path)
        if not recursive:
            raise IOError(f"{path} is a directory: pass recursive=True")
        for child in children:
            self.rm_file(child)
        self.invalidate_cache(self._parent(path))

    def touch(self, path, truncate: bool = True, **kwargs) -> None:
        path = self._strip_protocol(path)
        if truncate or path not in self.store.index.files:
            self.pipe_file(path, b"")

    # directories are virtual: they exist as soon as a file lives under them
    def mkdir(self, path, create_parents: bool = True, **kwargs) -> None:
        pass

    def makedirs(self, path, exist_ok: bool = False) -> None:
        pass

    def rmdir(self, path) -> None:
        path = self._strip_protocol(path)
        if self._is_known_dir(path):
            raise IOError(f"{path} is not empty")

    # -- file objects -----------------------------------------------------

    def _open(
        self,
        path,
        mode: str = "rb",
        block_size=None,
        autocommit: bool = True,
        cache_options=None,
        **kwargs,
    ):
        path = self._strip_protocol(path)
        if "x" in mode:
            if path in self.store.index.files:
                raise FileExistsError(path)
            mode = mode.replace("x", "w")
        if mode == "rb":
            return NXetFile(
                self,
                path,
                mode="rb",
                block_size=block_size,
                size=self._retry_when_missing(self.store.info, path).size,
                cache_options=cache_options,
                **kwargs,
            )
        if mode in ("wb", "ab"):
            if mode == "ab":
                try:
                    initial = self.store.read_file(path)
                except FileNotFoundError:
                    initial = b""
            else:
                initial = b""
            return NXetFile(
                self, path, mode="wb", block_size=block_size, initial_data=initial, **kwargs
            )
        raise NotImplementedError(f"nxet:// does not support mode {mode!r}")

    # -- nano-xet specifics -----------------------------------------------

    def header(self) -> dict:
        """The store header: format, hash and chunking parameters."""
        return self.store.header()

    def stats(self) -> Stats:
        """Deduplication statistics of the store."""
        return self.store.stats()

    def xorbs(self) -> List[dict]:
        """The physical objects of the store: name, size and number of chunks."""
        return [
            {
                "name": record.name,
                "size": record.size,
                "nchunks": len(record.chunks),
                "path": self.store.index.xorb_path(record.name),
            }
            for record in sorted(self.store.index.xorbs.values(), key=lambda r: r.name)
        ]

    def gc(self, dry_run: bool = False) -> List[str]:
        """Delete the xorbs that no file references any more."""
        return self.store.gc(dry_run=dry_run)

    def reload(self) -> None:
        """Re-read the index files (another writer may have updated the store)."""
        self.store.reload()

    def __repr__(self) -> str:
        return f"NXetFileSystem({PROTOCOL}://{self._uri_store})"


class NXetFile(AbstractBufferedFile):
    """File-like object over nano-xet chunks: random reads, buffered writes."""

    def __init__(self, fs, path, mode="rb", initial_data: bytes = b"", **kwargs):
        self._parts: List[bytes] = [initial_data] if initial_data else []
        super().__init__(fs, path, mode=mode, **kwargs)
        if mode != "rb" and initial_data:
            self.loc = len(initial_data)

    def _fetch_range(self, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        return self.fs.store.read_range(self.path, start, end)

    def _initiate_upload(self) -> None:
        pass

    def _upload_chunk(self, final: bool = False) -> bool:
        self.buffer.seek(0)
        data = self.buffer.read()
        if data:
            self._parts.append(data)
        if final:
            self.fs.store.write_file(self.path, b"".join(self._parts))
            self._parts = []
        return True


def _protocol_of(fs) -> str:
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    return protocol


fsspec.register_implementation(PROTOCOL, NXetFileSystem, clobber=True)
