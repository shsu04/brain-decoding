import os
import zstandard as zstd
from hashlib import sha256
import tarfile
import shutil
import tqdm
from multiprocessing import Pool, cpu_count


def _calculate_checksum(file_path: str):
    sha256_hash = sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _compress_single_directory(args):
    """Helper function to compress a single directory for parallel processing."""
    directory_path, output_path, compression_level = args
    try:
        cctx = zstd.ZstdCompressor(
            level=compression_level, threads=-1  # Use all available threads within zstd
        )
        zst_path = output_path + ".tar.zst"

        # Create a zstd compressed tar archive without creating an intermediate tar file
        with open(zst_path, "wb") as zst_out:
            with cctx.stream_writer(zst_out) as compressor:
                with tarfile.open(fileobj=compressor, mode="w|") as tar:
                    tar.add(directory_path, arcname=os.path.basename(directory_path))

        # Calculate checksum for the compressed file
        checksum = _calculate_checksum(zst_path)
        return (True, os.path.basename(directory_path), checksum)
    except Exception as e:
        return (False, os.path.basename(directory_path), str(e))


def compress_directories(
    base_path: str,
    destination_path: str,
    checksum_file_name: str,
    compression_level: int,
    num_workers: int = None,
):
    """Compresses all directories in the base_path into the destination_path.
    Saving checksums for later corruption verification. All non-directory files
    are copied to the destination directory as is.

    Parameters:
        base_path (str): The path to the base directory containing directories to compress.
        destination_path (str): The path where compressed files and checksums will be stored.
        checksum_file_name (str): The name of the checksum file.
        compression_level (int): The compression level for zstd.
        num_workers (int or str, optional): Number of worker processes to use. If 'mac', multiprocessing is disabled.

    """
    # Check if num_workers is 'mac', disable multiprocessing
    use_multiprocessing = True
    if num_workers == "mac":
        use_multiprocessing = False
    elif num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for other tasks

    directories = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]

    checksums = {}

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    if use_multiprocessing:
        # Prepare arguments for multiprocessing
        tasks = []
        for directory in sorted(directories):
            directory_path = os.path.join(base_path, directory)
            output_path = os.path.join(destination_path, directory)
            task = (directory_path, output_path, compression_level)
            tasks.append(task)

        # Compress in parallel using multiprocessing
        print(f"Compressing {len(tasks)} directories using {num_workers} processes...")
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(_compress_single_directory, tasks),
                    total=len(tasks),
                    desc="Compressing directories",
                )
            )
            pool.close()
            pool.join()
    else:
        # Single-threaded compression
        results = []
        for directory in tqdm.tqdm(sorted(directories), desc="Compressing directories"):
            directory_path = os.path.join(base_path, directory)
            output_path = os.path.join(destination_path, directory)
            success, dir_name, result = _compress_single_directory(
                (directory_path, output_path, compression_level)
            )
            results.append((success, dir_name, result))

    # Process results
    for success, directory_name, result in results:
        if success:
            checksums[directory_name] = result  # result is the checksum
        else:
            print(f"Failed to compress {directory_name}: {result}")

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

    print("Compression complete.")


def _decompress_single_file(args):
    """Helper function to decompress a single file for parallel processing."""
    tar_path, destination_path, directory_name, expected_checksum = args
    try:
        # Verify checksum
        actual_checksum = _calculate_checksum(tar_path)
        if expected_checksum != actual_checksum:
            return (False, os.path.basename(tar_path), "Checksum verification failed")

        # Decompress and extract using streams
        with open(tar_path, "rb") as compressed:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed) as reader:
                # Use tarfile in streaming mode
                with tarfile.open(fileobj=reader, mode="r|*") as tar:
                    tar.extractall(path=destination_path)
        return (True, os.path.basename(tar_path), "Success")
    except Exception as e:
        return (False, os.path.basename(tar_path), str(e))


def decompress_directories(
    source_path: str,
    destination_path: str,
    checksum_file_name: str,
    delete_compressed_files: bool = False,
    num_workers: int = None,
):
    """Decompresses directories from the source_path to the destination_path.
    Verifies checksums to detect any corruption.

    Parameters:
        source_path (str): The path containing compressed files and checksum file.
        destination_path (str): The path where decompressed files will be stored.
        delete_compressed_files (bool): Whether to delete the compressed files after decompression, if successful.
        checksum_file_name (str): The name of the checksum file.
        num_workers (int or str, optional): Number of worker processes to use. If 'mac', multiprocessing is disabled.

    """
    # Check if num_workers is 'mac', disable multiprocessing
    use_multiprocessing = True
    if num_workers == "mac":
        use_multiprocessing = False
    elif num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for other tasks

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

    if use_multiprocessing:
        # Prepare arguments for multiprocessing
        tasks = []
        for tar_file in sorted(tar_files):
            directory_name = tar_file.split(".")[0]
            tar_path = os.path.join(source_path, tar_file)
            expected_checksum = checksums.get(directory_name, None)
            if expected_checksum is None:
                print(f"No checksum found for {directory_name}, skipping.")
                continue
            task = (tar_path, destination_path, directory_name, expected_checksum)
            tasks.append(task)

        # Decompress in parallel using multiprocessing
        print(f"Decompressing {len(tasks)} files using {num_workers} processes...")
        with Pool(processes=num_workers) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap_unordered(_decompress_single_file, tasks),
                    total=len(tasks),
                    desc="Decompressing files",
                )
            )
            pool.close()
            pool.join()
    else:
        # Single-threaded decompression
        results = []
        for tar_file in tqdm.tqdm(sorted(tar_files), desc="Decompressing files"):
            directory_name = tar_file.split(".")[0]
            tar_path = os.path.join(source_path, tar_file)
            expected_checksum = checksums.get(directory_name, None)
            if expected_checksum is None:
                print(f"No checksum found for {directory_name}, skipping.")
                continue
            success, filename, message = _decompress_single_file(
                (tar_path, destination_path, directory_name, expected_checksum)
            )
            results.append((success, filename, message))

    # Process results
    for success, filename, message in results:
        if not success:
            corrupted_files.append((filename, message))

    # Copy non-compressed files to the destination
    for item in os.listdir(source_path):
        item_path = os.path.join(source_path, item)
        if (
            os.path.isfile(item_path)
            and not item.endswith(".tar.zst")
            and item != checksum_file_name
        ):
            destination_file_path = os.path.join(destination_path, item)
            shutil.copy2(item_path, destination_file_path)
            print(f"Copied {item} to {destination_path}")

    print("Decompression and verification complete")
    if corrupted_files:
        print("The following files failed:")
        for filename, error in corrupted_files:
            print(f"{filename}: {error}")
    else:
        print("All files decompressed successfully.")
        if delete_compressed_files:
            print("Deleting compressed files...")
            shutil.rmtree(source_path)
