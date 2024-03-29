import hashlib
import os
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Callable
from unittest import mock

from sister import download


class TestSetCacheRoot(unittest.TestCase):
    def test_set_cache_root(self):
        orig_root = download.get_cache_root()
        new_root = "/tmp/cache"
        try:
            download.set_cache_root(new_root)
            self.assertEqual(download.get_cache_root(), new_root)
        finally:
            download.set_cache_root(orig_root)


class TestGetCacheDirectory(unittest.TestCase):
    def test_get_cache_directory(self):
        root = download.get_cache_root()
        path = download.get_cache_directory("test", False)
        self.assertEqual(path, os.path.join(root, "test"))

    @mock.patch("os.makedirs")
    def test_fails_to_make_directory(self, f: Callable):
        f.side_effect = OSError()
        with self.assertRaises(OSError):
            download.get_cache_directory("/sister_test_cache", True)


class TestCachedDownload(unittest.TestCase):
    def setUp(self):
        self.default_cache_root = download.get_cache_root()
        self.temp_dir = tempfile.mkdtemp()
        download.set_cache_root(self.temp_dir)

    def tearDown(self):
        download.set_cache_root(self.default_cache_root)
        shutil.rmtree(self.temp_dir)

    @mock.patch("os.makedirs")
    def test_fails_to_make_directory(self, f: Callable):
        f.side_effect = OSError()
        with self.assertRaises(OSError):
            download.cached_download("https://example.com")

    def test_file_exists(self):
        # Make an empty file which has the same name as the cache directory
        with open(os.path.join(self.temp_dir, "_dl_cache"), "w"):
            pass
        with self.assertRaises(OSError):
            download.cached_download("https://example.com")

    @mock.patch("os.path.exists")
    def test_cache_exists(self, f: Callable):
        f.return_value = True
        url = "https://example.com"
        path = download.cached_download(url)
        self.assertEqual(
            path,
            os.path.join(
                self.temp_dir, "_dl_cache", hashlib.md5(url.encode("utf-8")).hexdigest()
            ),
        )

    @mock.patch("urllib.request.urlretrieve")
    def test_cached_download(self, f: Callable):
        def urlretrieve(url, path, progress_hook=None):
            with open(path, "w") as f:
                f.write("test")

        f.side_effect = urlretrieve

        cache_path = download.cached_download("https://example.com")

        self.assertEqual(f.call_count, 1)
        args, kwargs = f.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(len(args), 3)
        # The second argument is a temporary path, and it is removed
        self.assertEqual(args[0], "https://example.com")

        self.assertTrue(os.path.exists(cache_path))
        with open(cache_path) as f:
            stored_data = f.read()
        self.assertEqual(stored_data, "test")


class CachedUnzipCase(unittest.TestCase):
    def setUp(self):
        tempfp = tempfile.NamedTemporaryFile()
        tempfilepath = tempfp.name
        temp_dir = tempfile.mkdtemp()
        zippath = os.path.join(temp_dir, "test.zip")
        # Create temp zip file
        zipfile.ZipFile(zippath, "w").write(tempfilepath)

        self.tempfp = tempfp
        self.tempfilepath = tempfilepath
        self.temp_dir = temp_dir
        self.zippath = zippath

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_cache_unzip(self):
        # TODO: Not tested if the zipfile is actually unziped.
        saveto = Path(self.temp_dir) / "saveto"
        download.cached_unzip(Path(self.zippath), saveto=saveto)
