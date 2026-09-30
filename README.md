# nano-xet

A tiny, readable re-implementation of the ideas behind [Xet](https://huggingface.co/docs/xet/en/index)
— the content-defined-chunking storage layer Hugging Face uses for large files — in
~1000 lines of Python, on top of [fsspec](https://filesystem-spec.readthedocs.io).

Write a file to `nxet://`, and it is cut into content-defined chunks, hashed, deduplicated
against what is already stored, packed into **xorb** objects (each xorb holds many chunks),
and recorded in small JSON index files under `.nxet/` on any filesystem fsspec knows about.

It is a teaching/demo tool, **not** a production storage system: no compression, no
encryption, no CAS server, no concurrent writers. Target size: files below ~300 MB.

```bash
export FSSPEC_NXET_STORE_URI=/Users/me/tmp/my-nxet-store   # where the xorbs live
```

```python
import fsspec
with fsspec.open("nxet://my/path/to/data.csv", "wb") as f:   # nxet:// + the store above
    f.write(b"hello nano-xet\n")
```

Without the environment variable, name the store inline after `::` — the two halves of the
URI are the file you see, and where its chunks end up:

```python
with fsspec.open("nxet://my/path/to/data.csv::file:///Users/me/tmp/my-nxet-store", "rb") as f:
    print(f.read())
```

## Install

```bash
pip install nano-xet            # or: pip install "nano-xet[fast]" for numpy chunking
```

## Quick start

### fsspec

```python
import fsspec

fs = fsspec.filesystem("nxet", store_uri="file:///Users/me/tmp/my-nxet-store")
# or fsspec.filesystem("nxet", fo="/Users/me/tmp/my-nxet-store", target_protocol="file")
# or export FSSPEC_NXET_STORE_URI=... and use fsspec.filesystem("nxet") / fsspec.open("nxet://...")

fs.pipe_file("data/train.csv", b"id,value\n0,1\n1,2\n")
fs.cat_file("data/train.csv")                  # b'id,value\n0,1\n1,2\n'
fs.ls("data")                                  # virtual dirs, nothing on disk
fs.cat_file("data/train.csv", start=9)         # random access to any byte range
fs.pipe_file("data/train_v2.csv", new_bytes)   # only new chunks are written
fs.stats()                                     # deduplication statistics
```

Everything fsspec can do works: `find`, `glob`, `walk`, `copy`, `mv`, `get`, `put`,
`tail`, `head`, `touch`, append mode, text mode, and chained URIs against other
protocols (`memory://`, `s3://`, `gs://`, `smb://`, …).

### Store API (no fsspec needed)

```python
from nano_xet import NXetStore

with NXetStore.open("file:///Users/me/tmp/my-nxet-store") as store:
    store.write_file("data/train.csv", data)
    store.read_range("data/train.csv", 1_000_000, 1_000_128)
    print(store.stats().summary())
```

```text
1 file(s), 1681 chunk(s), 1 xorb(s)
logical : 106.7 MB
stored  : 53.5 MB (842 unique chunk(s))
dedup   : 53.2 MB saved (49.9%) [839 chunk(s) reused]
```

### CLI

```bash
export FSSPEC_NXET_STORE_URI=/tmp/my-store        # one store per shell

nxet put train.csv nxet://data/train.csv
nxet put train.csv nxet://data/train_v2.csv       # dedup: only new chunks land
nxet ls -R nxet://data
nxet cat nxet://data/train.csv --start 0 --end 100
nxet get nxet://data/train.csv ./train.csv
nxet stats
nxet xorbs
nxet gc
```

Every command also takes the store inline, which is handy in scripts:
`nxet stats nxet://::file:///tmp/my-store`.

### Demo

```bash
python examples/demo.py /tmp/nano-xet-demo
```

Writes two versions of an 11 MB CSV and shows that the second one costs 0.6 MB.

## How it works

```
                          data.csv
                             │
                    gear hash (same table as Xet)
                             ▼
                chunks of 8 KiB … 64 KiB … 128 KiB
                             │
                        blake2b-256
                             ▼
                chunk hash already known?                    ┌────────────────────────┐
                    │                        ┌── yes ───────▶│ .nxet/nxet.json        │
                    no                                       │  format, chunk sizes,  │
                    ▼                                        │  xorb counter          │
     buffered in the current xorb                            └────────────────────────┘
                    │ xorb full: 64 MiB or 8192 chunks
                    ▼                                        ┌────────────────────────┐
     ┌───────────────────────────────┐                       │ .nxet/nxet.files.jsonl │
     │ 000000-a1b2c3….xorb           │ ◀── chunks sorted by  │  path → [(hash, size)] │
     │ chunk… │ chunk… │ chunk… │   │     hash, raw bytes   └────────────────────────┘
     └───────────────────────────────┘                                  │
                    ▲                                                   ▼
                    └───────── read: hash → (xorb, offset) ────  ┌────────────────────────┐
                                                                 │ .nxet/nxet.xorbs.jsonl │
                                                                 │  hash → xorb + offset  │
                                                                 └────────────────────────┘
```

1. **Chunking** (`chunking.py`) — content-defined chunking with the **same gear hash, the
   same 256-entry lookup table, the same boundary mask and the same size limits as Xet**:
   64 KiB mean, 8 KiB minimum, 128 KiB maximum, boundary when `hash & mask == 0`.
   Chunk boundaries therefore match what `xet-core` produces for the same input
   (`tests/test_chunking.py` compares against golden values generated by a Rust
   reference chunker).
2. **Hashing** (`hashing.py`) — every chunk is identified by its `blake2b-256` digest, used
   as the deduplication key (`blake3` from the standard library equivalent, no dependency).
3. **Xorbs** (`store.py`) — chunks are appended to the current xorb until it reaches
   64 MiB or 8192 chunks (Xet's own limits), then a new xorb starts. Chunks inside a xorb
   are sorted by hash and stored raw, so a xorb is a plain concatenation of chunk bytes
   and a chunk is read with a single `pread`.
4. **Index** (`index.py`) — three small JSON/JSONL files under `.nxet/`:
   `nxet.json` (header), `nxet.files.jsonl` (append-only `put`/`rm` records: path → chunk
   hashes), `nxet.xorbs.jsonl` (xorb → `[hash, size, offset]` per chunk). Directories are
   virtual: they exist because some file path has them as a prefix.

A file is a list of chunk hashes; reading is `hash → (xorb, offset, size) → bytes`, and
consecutive chunks in the same xorb are coalesced into one read.

### What ends up on disk

Only xorbs sit at the root; the link files are hidden in `.nxet/`:

```text
my-nxet-store/
├── 000000-a1b2c3d4e5f6a7b8.xorb     000001-9c0d1e2f3a4b5c6d.xorb   … many chunks each
└── .nxet/
    ├── nxet.json                    format, hash algorithm, chunk sizes, xorb counter
    ├── nxet.files.jsonl             put/rm records: path -> [(chunk hash, size)]
    └── nxet.xorbs.jsonl             xorb -> [(chunk hash, size, offset)]
```

They are plain JSON, so a store can be inspected with `jq` and `grep`:

```bash
grep -c . .nxet/nxet.files.jsonl                        # one line per write and delete
jq '.size, .chunks | length' <(head -1 .nxet/nxet.files.jsonl)
du -sh . ; du -sh .nxet                                 # data vs. index
```

### Same as Xet / different from Xet

| | nano-xet | Xet |
|---|---|---|
| gear hash table, boundary mask, min/mean/max chunk size | identical | — |
| `xorb` as a physical multi-chunk container, 64 MiB / 8192 chunks | identical | — |
| chunk deduplication across files and versions | yes | yes |
| hashing | `blake2b-256` | keyed `blake3` |
| xorb content | raw chunk bytes | byte-grouped, compressed, encrypted |
| index | JSON/JSONL under `.nxet/` | sharded merkle tables in a CAS |
| metadata updates | last write wins, `reload()` to see others | CAS + commit with rebase |
| storage | any fsspec filesystem | HF CAS (+ local cache) |
| language | Python | Rust |

## Performance

On an Apple M-series laptop, Python 3.12, a 107 MB CSV (1681 chunks, mean 64 KiB):

| operation | cost |
|---|---|
| chunking | 2.2 s with numpy (49 MB/s), 11 s pure Python (10 MB/s) |
| write: chunk + hash + dedup + store | 2.4 s |
| write a second version with 1 byte inserted | 2.4 s, +1 chunk stored |
| read the whole file back | 0.05 s (~2 GB/s from OS cache) |
| 20 random 1 KB reads at different offsets | 1.1 ms |
| two identical 107 MB files | 107 MB stored, not 214 MB |

The numpy path is optional (`use_numpy=False`, or the `fast` extra) and gives ~4x faster
chunking; the pure Python path keeps nano-xet dependency-free apart from fsspec.

## Limitations (by design)

- One writer at a time. Readers pick up other writers' changes on the next miss
  (`reload_if_stale`), but two processes writing at the same time can lose a file record.
- `.nxet/` and `*.xorb` are reserved at the root of the store: writing such a path is an
  error rather than a silent collision.
- No compression, no encryption, no partial-file corruption recovery: a xorb is raw bytes.
- Everything needed to rebuild a file is in the JSONL index, so huge datasets mean big
  index files (nano-xet is meant for ≤ 300 MB files, not for a whole repository).
- `gc` must be run when files are deleted; unreferenced xorbs are only garbage (`nxet stats`
  tells you how much is waiting to be collected).
- Chunking is Xet-compatible, but the file *hash* is not a Xet merkle hash, so nano-xet
  stores and Xet stores are not interchangeable.
- `memory://` works as an underlying filesystem for tests and demos, but it is per-process:
  two `nxet` commands do not share it.

## Development

```bash
pip install -e ".[test,fast]"
pytest -q                     # 168 passed, 1 xfailed
python examples/demo.py
ruff check src tests examples
```

Sources of truth for the parts copied from Xet:
[`xet-core`](https://github.com/huggingface/xet-core) —
`xet_data/src/deduplication/chunking.rs` (chunker) and
`xet_core_structures/src/xorb_object/constants.rs` (xorb/chunk sizes), plus the
`gearhash` crate for the lookup table.

## License

Apache-2.0. See [LICENSE](LICENSE).
