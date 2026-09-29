import pytest

from nano_xet import NXetFileSystem, NXetStore


@pytest.fixture
def store_path(tmp_path):
    return tmp_path / "store"


@pytest.fixture
def store(store_path):
    with NXetStore.open(f"file://{store_path}") as store:
        yield store


@pytest.fixture
def fs(store_path):
    return NXetFileSystem(fo=str(store_path), target_protocol="file")


@pytest.fixture
def csv_data():
    """~600 KB of csv text: several chunks, and stable under appends."""
    lines = ["a,b,c"]
    for i in range(12000):
        lines.append(f"{i},value-{i % 97},{i * 37 % 1000}")
    return "\n".join(lines).encode()


@pytest.fixture
def binary_data():
    import random

    return random.Random(1234).randbytes(400_000)


# fsspec's own abstract copy/get suite has one test that cannot pass here:
# other_paths() only rebuilds the destination tree of a recursive get() into an
# existing directory for filesystems whose paths start with "/" (memory:// does,
# s3:// and nxet:// keys do not). Everything else in the suite is run as-is.
XFAILS = {
    "test_get_directory_recursive": (
        "fsspec other_paths() drops the source root when the destination is an "
        "existing directory and remote paths have no leading '/'"
    ),
}


def pytest_collection_modifyitems(items):
    for item in items:
        reason = XFAILS.get(item.name)
        if reason:
            item.add_marker(pytest.mark.xfail(reason=reason, strict=True))
