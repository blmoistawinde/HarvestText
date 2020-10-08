import os
import shutil
import requests
import hashlib
from tqdm import tqdm
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext

RemoteFileMetadata = namedtuple('RemoteFileMetadata',
                                ['filename', 'url', 'checksum'])

# config according to computer, this should be default setting of shadowsocks
DEFAULT_PROXIES = {
    'http': 'socks5h://127.0.0.1:1080',
    'https': 'socks5h://127.0.0.1:1080'
}

def get_data_home(data_home=None):
    """Return the path of the scikit-learn data dir.
    This folder is used by some large dataset loaders to avoid downloading the
    data several times.
    By default the data dir is set to a folder named 'scikit_learn_data' in the
    user home folder.
    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str | None
        The path to data dir.
    """
    if data_home is None:
        data_home = environ.get('HARVESTTEXT_DATA',
                                join('~', '.harvesttext'))
    data_home = expanduser(data_home)
    if not exists(data_home):
        makedirs(data_home)
    return data_home

def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.
    Parameters
    ----------
    data_home : str | None
        The path to data dir.
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)

def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while True:
            buffer = f.read(chunk_size)
            if not buffer:
                break
            sha256hash.update(buffer)
    return sha256hash.hexdigest()

def _download_with_bar(url, file_path, proxies=DEFAULT_PROXIES):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True, proxies=proxies)
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024    # 1 KB
    progress_bar = tqdm(total=total_size_in_bytes, unit='B', unit_scale=True)
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise Exception("ERROR, something went wrong with the downloading")
    return file_path

def _fetch_remote(remote, dirname=None, use_proxy=False, proxies=DEFAULT_PROXIES):
    """Helper function to download a remote dataset into path
    Fetch a dataset pointed by remote's url, save into path using remote's
    filename and ensure its integrity based on the SHA256 Checksum of the
    downloaded file.
    Parameters
    ----------
    remote : RemoteFileMetadata
        Named tuple containing remote dataset meta information: url, filename
        and checksum
    dirname : string
        Directory to save the file to.
    Returns
    -------
    file_path: string
        Full path of the created file.
    """

    file_path = (remote.filename if dirname is None
                 else join(dirname, remote.filename))
    proxies = None if not use_proxy else proxies
    file_path = _download_with_bar(remote.url, file_path, proxies)
    checksum = _sha256(file_path)
    if remote.checksum != checksum:
        raise IOError("{} has an SHA256 checksum ({}) "
                      "differing from expected ({}), "
                      "file may be corrupted.".format(file_path, checksum,
                                                      remote.checksum))
    return file_path


def download(remote, file_path=None, use_proxy=False, proxies=DEFAULT_PROXIES):
    data_home = get_data_home()
    file_path = _fetch_remote(remote, data_home, use_proxy, proxies)
    return file_path

def check_download_resource(remote, use_proxy=False, proxies=None):
    proxies = DEFAULT_PROXIES if use_proxy and proxies is None else proxies
    data_home = get_data_home()
    file_path = os.path.join(data_home, remote.filename)
    if not os.path.exists(file_path):
        # currently don't capture error at this level, assume download success
        file_path = download(remote, data_home)
    return file_path

if __name__ == "__main__":
    ARCHIVE = RemoteFileMetadata(
        filename='harvesttext-0.7.2-py3-none-any.whl',
        url='https://github.com/blmoistawinde/HarvestText/releases/download/V0.7.2/harvesttext-0.7.2-py3-none-any.whl',
        checksum='004c8b0b1858f69025f721bc84cff33127d53c6ab526beed7a7a801a9c21f30b')
    print("Download")
    file_path = download(ARCHIVE)
    print(file_path)
    # if proxy is available
    # print("Download using proxy")
    # file_path = download(ARCHIVE, use_proxy=True)
    # print(file_path)