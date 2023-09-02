import os
import argparse
import zipfile
import wget


def download(url, targetdir):
    """
    Download a file and save it in some target directory.

    Args:
        url: The url from which the file must be downloaded.
        targetdir: The path to the directory where the file must be saved.

    Returns:
        The path to the downloaded file.
    """
    print("* Downloading data from {}...".format(url))
    filepath = os.path.join(targetdir, url.split('/')[-1])
    wget.download(url, filepath)
    return filepath


def unzip(names, filepath):
    """
    Extract the data from a zipped file and delete the archive.

    Args:
        filepath: The path to the zipped file.
    """
    print("\n* Extracting: {}...".format(filepath))
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        if names == None:
            for name in zf.namelist():
                # Ignore useless files in archives.
                if "__MACOSX" in name or \
                        ".DS_Store" in name or \
                        "Icon" in name:
                    continue
                zf.extract(name, dirpath)
        else:
            zf.extract(names, filepath)
    # Delete the archive once the data has been extracted.
    os.remove(filepath)


def download_unzip(url, name, targetdir):
    """
    Download and unzip data from some url and save it in a target directory.

    Args:
        url: The url to download the data from.
        targetdir: The target directory in which to download and unzip the
                   data.
    """
    filepath = os.path.join(targetdir, url.split('/')[-1])
    target = os.path.join(targetdir,
                          ".".join((url.split('/')[-1]).split('.')[:-1]))

    if not os.path.exists(targetdir):
        print("* Creating target directory {}...".format(targetdir))
        os.makedirs(targetdir)

    # Skip download and unzipping if the unzipped data is already available.
    if os.path.exists(target) or os.path.exists(target + ".txt"):
        print("* Found unzipped data in {}, skipping download and unzip..."
              .format(targetdir))
    # Skip downloading if the zipped data is already available.
    elif os.path.exists(filepath):
        print("* Found zipped data in {} - skipping download..."
              .format(targetdir))
        unzip(name, filepath)
    # Download and unzip otherwise.
    else:
        unzip(name, download(url, targetdir))


if __name__ == "__main__":
    # Default data.
    # COILL dataset need to be download from Reuters.
    # Glove word  embeddings(6B,100d)
    glove_url = "http://www-nlp.stanford.edu/data/glove.6B.zip"
    parser = argparse.ArgumentParser(description='Download the COILL dataset')
    parser.add_argument("--embeddings_url",
                        default=glove_url,
                        help="URL of the pretrained embeddings to download")
    parser.add_argument("--target_dir",
                        default=os.path.join(".", "dataset"),
                        help="Path to a directory where data must be saved")
    args = parser.parse_args()

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    print(20 * "=", "Fetching the word embeddings:", 20 * "=")
    download_unzip(args.embeddings_url,None,
                   os.path.join(args.target_dir, "embeddings"))
