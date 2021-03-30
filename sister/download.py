import bz2
import contextlib
import gzip
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib import request

import progressbar

_cache_root = os.environ.get(
    "SISTER_CACHE_ROOT", os.path.join(os.path.expanduser("~"), ".sister", "cache")
)


@contextlib.contextmanager
def tempdir(**kwargs):
    ignore_errors = kwargs.pop("ignore_errors", False)

    temp_dir = tempfile.mkdtemp(**kwargs)
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=ignore_errors)


def get_cache_root() -> str:
    return _cache_root


def set_cache_root(path: str) -> None:
    global _cache_root
    _cache_root = path


def get_cache_directory(cache_name: str, create_directory: bool = True) -> str:
    path = os.path.join(_cache_root, cache_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


def show_progress(block_num, block_size, total_size):
    pbar = progressbar.ProgressBar(maxval=total_size)
    pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def cached_download(url: str) -> str:
    cache_root = os.path.join(_cache_root, "_dl_cache")
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.isdir(cache_root):
            raise

    urlhash = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_root, urlhash)

    if os.path.exists(cache_path):
        return cache_path

    with tempdir(dir=cache_root) as temp_root:
        temp_path = os.path.join(temp_root, "dl")
        sys.stderr.write("Downloading from {}...\n".format(url))
        sys.stderr.flush()
        request.urlretrieve(url, temp_path, show_progress)
        shutil.move(temp_path, cache_path)

    return cache_path


def cached_unzip(path: Path, saveto: Path) -> None:
    if not saveto.exists():
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(saveto)


def cached_decompress_bz2(path: Path, saveto: Path) -> None:
    if not saveto.exists():
        with bz2.open(path, "rb") as fin:
            _content = fin.read()
        with open(saveto, "wb") as fout:
            fout.write(_content)


def cached_decompress_gzip(path: Path, saveto: Path) -> None:
    if not saveto.exists():
        with gzip.open(path, "rb") as fin:
            _content = fin.read()
        with open(saveto, "wb") as fout:
            fout.write(_content)
