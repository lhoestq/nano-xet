"""nano-xet demo: two versions of a dataset, stored once.

    python examples/demo.py                # uses a temporary directory
    python examples/demo.py /tmp/my-store  # or a store you keep

Every step prints what it does; the interesting part is the statistics at the
end: version 2 shares almost all of its chunks with version 1, so writing it
costs almost nothing.
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile

import fsspec

from nano_xet import NXetFileSystem, NXetStore


def make_rows(count: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    return "".join(
        f"{i},{rng.choice(['a', 'b', 'c'])}-{i % 997},{rng.random():.8f},{i * 7919 % 1000}\n"
        for i in range(count)
    ).encode()


def main(store_dir: str = "") -> None:
    keep = bool(store_dir)
    store_dir = store_dir or tempfile.mkdtemp(prefix="nano-xet-demo-")
    underlying = f"file://{store_dir}"
    fs = NXetFileSystem(fo=store_dir, target_protocol="file")

    v1 = make_rows(400_000)
    half = len(v1) // 2
    v2 = v1[:half] + b"# a new column appeared here\n" + v1[half:] + make_rows(20_000, seed=1)
    print(f"dataset v1: {len(v1) / 1e6:.1f} MB   v2: {len(v2) / 1e6:.1f} MB")

    # --- write both versions, through plain fsspec ------------------------
    with fsspec.open(f"nxet://data/users-v1.csv::{underlying}", "wb") as f:
        f.write(v1)
    fs.pipe_file("data/users-v2.csv", v2)  # same thing with the filesystem object
    print("wrote nxet://data/users-v1.csv and nxet://data/users-v2.csv\n")

    # --- what the store looks like -----------------------------------------
    print(f"nxet://::file://{store_dir}")
    for entry in sorted(fs.find("", detail=True).values(), key=lambda e: e["name"]):
        print(f"  {entry['name']:>22}  {entry['size'] / 1e6:6.1f} MB  {entry['nchunks']} chunks")
    print("\nphysical xorbs (each one holds many chunks):")
    for xorb in fs.xorbs():
        print(f"  {xorb['path']}\n    {xorb['size'] / 1e6:6.1f} MB, {xorb['nchunks']} chunks")

    # --- random access ---------------------------------------------------
    with fsspec.open(f"nxet://data/users-v2.csv::{underlying}", "rb") as f:
        first_line = f.readline()
        f.seek(1_000_000)
        middle = f.read(48)
    assert first_line == v2.split(b"\n", 1)[0] + b"\n"
    assert middle == v2[1_000_000 : 1_000_048]
    print(f"\nrandom access: first line {first_line!r}\n               bytes @1MB {middle!r}")

    # --- the point of it all --------------------------------------------
    print(f"\n{fs.stats().summary()}")

    # --- the same store, without fsspec ----------------------------------
    with NXetStore.open(f"nxet://::{underlying}") as store:
        assert store.read_file("data/users-v1.csv") == v1
        assert store.read_file("data/users-v2.csv") == v2
    print("\nNXetStore.open() reads the same store back without fsspec")

    if keep:
        print(f"\nstore kept in {store_dir}\ntry: nxet ls -R nxet://::file://{store_dir}")
    else:
        shutil.rmtree(store_dir, ignore_errors=True)
        print(f"\n(temporary store {store_dir} removed)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
