"""``nxet`` command line: put/get/ls/cat/rm files in a store, and look at the dedup.

The store is always given as a full chained uri::

    nxet put train.csv nxet://data/train.csv::file:///tmp/my-nxet-store
    nxet ls nxet://::file:///tmp/my-nxet-store -R
    nxet stats nxet://::file:///tmp/my-nxet-store
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from typing import List, Optional

import fsspec

from .store import human_bytes


def _resolve(uri: str):
    fs, path = fsspec.url_to_fs(uri)
    return fs, path


def split_chain(uri: str) -> List[str]:
    return uri.split("::")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nxet", description="nano-xet: deduplicated files on any filesystem"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str, uri_optional: bool = False):
        p = sub.add_parser(name, help=help_text)
        kwargs = {"nargs": "?", "default": "nxet://"} if uri_optional else {}
        p.add_argument("uri", help="nxet://[path]::<underlying uri>", **kwargs)
        return p

    put = sub.add_parser("put", help="copy local file(s) into the store")
    put.add_argument("local", nargs="+")
    put.add_argument("uri", help="nxet://[dir/]::(<underlying uri>) destination")

    get = sub.add_parser("get", help="copy file(s) out of the store")
    get.add_argument("uri", help="nxet://<path>::<underlying uri> (or directory with -r)")
    get.add_argument("local", help="local file or directory")
    get.add_argument("-r", "--recursive", action="store_true")

    ls = sub.add_parser("ls", help="list the store")
    ls.add_argument("uri", nargs="?", default="nxet://")
    ls.add_argument("-R", "--recursive", action="store_true")

    cat = add("cat", "print a file")
    cat.add_argument("--start", type=int, default=None, help="first byte to print")
    cat.add_argument("--end", type=int, default=None, help="end of the byte range")
    rm = sub.add_parser("rm", help="delete a file or directory")
    rm.add_argument("uri")
    rm.add_argument("-r", "--recursive", action="store_true")

    add("stats", "deduplication statistics of the store", uri_optional=True)
    add("xorbs", "list the physical xorb objects", uri_optional=True)

    gc = sub.add_parser("gc", help="delete the xorbs that no file references")
    gc.add_argument("uri", nargs="?", default="nxet://")
    gc.add_argument("--dry-run", action="store_true")

    add("head", "show the store header (chunking parameters)", uri_optional=True)

    args = parser.parse_args(argv)
    command = args.command

    if command == "put":
        fs, path = _resolve(args.uri)
        target = path.rstrip("/")
        raw_target = split_chain(args.uri)[0].rstrip()
        many = len(args.local) > 1
        for local in args.local:
            name = local.rsplit("/", 1)[-1]
            is_dir = not target or fs.isdir(target) or raw_target.endswith("/")
            if many or is_dir:
                name = f"{target}/{name}".strip("/")
            else:
                name = target
            with open(local, "rb") as src:
                fs.pipe_file(name, src.read())
            print(f"{local} -> {fs.unstrip_protocol(name)}")
    elif command == "get":
        fs, path = _resolve(args.uri)
        if args.recursive or fs.isdir(path):
            fs.get(path, args.local, recursive=True)
            print(f"nxet://{path} -> {args.local}")
        else:
            with fs.open(path, "rb") as src, open(args.local, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"nxet://{path} -> {args.local}")
    elif command == "ls":
        fs, path = _resolve(args.uri)
        entries = fs.find(path, detail=True) if args.recursive else fs.ls(path, detail=True)
        entries = list(entries.values()) if isinstance(entries, dict) else list(entries)
        for entry in sorted(entries, key=lambda e: e["name"]):
            kind = "d" if entry["type"] == "directory" else "-"
            size = human_bytes(entry["size"]) if entry["size"] is not None else ""
            print(f"{kind} {size:>10} {entry['name']}")
    elif command == "cat":
        fs, path = _resolve(args.uri)
        sys.stdout.buffer.write(fs.cat_file(path, start=args.start, end=args.end))
    elif command == "head":
        fs, _ = _resolve(args.uri)
        print(json.dumps(fs.header(), indent=2))
    elif command == "rm":
        fs, path = _resolve(args.uri)
        fs.rm(path, recursive=args.recursive)
        print(f"deleted nxet://{path}")
    elif command == "stats":
        fs, _ = _resolve(args.uri)
        print(fs.stats().summary())
    elif command == "xorbs":
        fs, _ = _resolve(args.uri)
        for xorb in fs.xorbs():
            print(
                f"{xorb['name']}  {human_bytes(xorb['size']):>10}  "
                f"{xorb['nchunks']:>6} chunk(s)"
            )
    elif command == "gc":
        fs, _ = _resolve(args.uri)
        removed = fs.gc(dry_run=args.dry_run)
        for name in removed:
            print(("would delete " if args.dry_run else "deleted ") + name)
        if not removed:
            print("nothing to delete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
