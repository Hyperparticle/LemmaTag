#!/usr/bin/env python3

# Downloads universal dependencies datasets
# See http://universaldependencies.org for a list of all datasets


from tqdm import tqdm
import requests
import math
import tarfile


def download_file(url, filename=None):
    # Streaming, so we can iterate over the response
    r = requests.get(url, stream=True)

    # Total size in bytes
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print("Error, something went wrong")


def extract_file(read_filename, output_path):
    tar = tarfile.open(read_filename, 'r')
    tar.extractall(output_path)


if __name__ == "__main__":
    import os

    data_folder = "data"
    dataset_url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz" \
                  "?sequence=1&isAllowed=y"
    dataset_path = os.path.join(data_folder, "ud-treebanks-v2.2.tgz")

    print("Downloading dataset")
    download_file(dataset_url, dataset_path)

    print("Extracting dataset")
    extract_file(dataset_path, data_folder)

    print("Completed successfully")
