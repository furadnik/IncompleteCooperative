"""Helper script for finding data.json files in a path."""
from pathlib import Path
from sys import stderr
from typing import Iterator

from incomplete_cooperative.run.save import Output, get_outputs_from_file


def find_data_jsons(path: Path) -> Iterator[Path]:
    """Find all data.json files in the subdirectories."""
    if (path / "data.json").exists():
        print(path, file=stderr)
        yield path / "data.json"
        return

    for x in path.iterdir():
        if x.is_dir():
            yield from find_data_jsons(x)


def find_data(path: Path) -> Iterator[dict[str, Output]]:
    """Find all data.jsons in path, return their parsed outputs."""
    return map(get_outputs_from_file, find_data_jsons(path))
