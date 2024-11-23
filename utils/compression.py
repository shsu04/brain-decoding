import os
import zstandard as zstd
from hashlib import sha256
import time
import tarfile
import shutil
from io import BytesIO
import tqdm


def _calculate_checksum(file_path: str):
    sha256_hash = sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _compress_directory_with_zstd(
    directory_path: str, output_path: str, compression_level: int
):
    cctx = zstd.ZstdCompressor(
        level=compression_level, threads=-1
    )  # Use all available threads
    tar_path = output_path + ".tar"
    zst_path = output_path + ".tar.zst"

    # Create a tar file for the directory
    with tarfile.open(tar_path, "w") as tar:
        tar.add(directory_path, arcname=os.path.basename(directory_path))

    # Compress the tar file with zstd
    with open(tar_path, "rb") as tar, open(zst_path, "wb") as zst:
        cctx.copy_stream(tar, zst)

    # Remove the uncompressed tar file
    os.remove(tar_path)

    # Calculate checksum for the compressed file
    return _calculate_checksum(zst_path)


def compress_directories(
    base_path: str,
    destination_path: str,
    checksum_file_name: str,
    compression_level: int,
):
    """Compresses all directories in the base_path into the destination_path
    Saving checksums for later corruption verification. All non-directory files
    are copied to the destination directory as is.
    """

    directories = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    checksums = {}

    for directory in tqdm.tqdm(sorted(directories)):
        directory_path = os.path.join(base_path, directory)
        output_path = os.path.join(destination_path, directory)

        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        checksums[directory] = _compress_directory_with_zstd(
            directory_path, output_path, compression_level
        )

    # Save checksums to a file for later verification
    with open(os.path.join(destination_path, checksum_file_name), "w") as f:
        for directory, checksum in checksums.items():
            f.write(f"{directory}: {checksum}\n")

    # Copy remaining files to the destination directory
    for file in os.listdir(base_path):
        item_path = os.path.join(base_path, file)
        if os.path.isfile(item_path):
            destination_file_path = os.path.join(destination_path, file)
            shutil.copy2(item_path, destination_file_path)
            print(f"Copied {file} to {destination_path}")

    return


def decompress_directories(source_path: str, destination_path, checksum_file_name: str):
    with open(os.path.join(source_path, checksum_file_name), "r") as f:
        checksums = {
            line.split(": ")[0]: line.split(": ")[1].strip() for line in f.readlines()
        }

    tar_files = [
        f
        for f in os.listdir(source_path)
        if os.path.isfile(os.path.join(source_path, f)) and f.endswith(".tar.zst")
    ]
    corrupted_files = []

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    for tar_file in tqdm.tqdm(sorted(tar_files)):
        directory_name = tar_file.split(".")[0]
        tar_path = os.path.join(source_path, tar_file)

        # Verify checksum
        checksum = _calculate_checksum(tar_path)
        if checksums.get(directory_name, None) == checksum:
            # Decompress using the zstd module
            with open(tar_path, "rb") as compressed:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    tar_content = BytesIO(reader.read())
                    with tarfile.open(fileobj=tar_content, mode="r:") as tar:
                        tar.extractall(path=destination_path)
        else:
            corrupted_files.append(tar_file)

    # Copy non-tar.zst files to the destination
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        # Check if it is a file and does not end with '.tar.zst'
        if os.path.isfile(item_path) and not item.endswith(".tar.zst"):
            destination_file_path = os.path.join(destination_path, item)
            shutil.copy2(item_path, destination_file_path)
            print(f"Copied {item} to {destination_path}")

    print("Decompression and verification complete")
    if corrupted_files:
        print("The following files are corrupted:")
        for file in corrupted_files:
            print(file)
