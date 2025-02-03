import multiprocessing
import sys
import os
import threading
import time
import pexpect
import hashlib
import concurrent.futures
import osfclient
import tqdm
import os

"""
Utilities for downloading from repositories and verification.
"""


def download_radboud(
    root_dir: str,
    base_url: str,
    repo_path: str,
):
    """Downloads from the Radboud Data Repository."""

    # Set downloader by platform
    if sys.platform.startswith("darwin"):
        downloader = "environments/repocli.darwin_arm64"
    elif sys.platform.startswith("linux"):
        downloader = "environments/repocli.x86_64"
    else:
        raise NotImplementedError("Only Linux and MacOS are supported")

    assert os.path.exists(downloader), f"Downloader not found: {downloader}"

    # Configure
    child = pexpect.spawn(f"{downloader} config")

    username = input("Enter Radboud Data Repository username: ")
    password = input("Enter Radboud Data Repository password: ")

    while True:
        idx = child.expect(
            [
                r"login for",  # if configured base
                r"repo baseurl",  # if not
                r"username:",
                r"password:",
                pexpect.EOF,
                pexpect.TIMEOUT,
            ],
            timeout=10,
        )

        # Informational, ignore.
        if idx == 0:
            pass
        elif idx == 1:
            child.sendline(base_url)
        elif idx == 2:
            child.sendline(username)
        elif idx == 3:
            child.sendline(password)
        elif idx == 4:
            break
        elif idx == 5:
            print("Timed out waiting for repocli config prompt.")
            break

    # Print anything left in the buffer
    output = child.before.decode(errors="replace")
    print(output)
    child.close()
    print(f"repocli config exited with status {child.exitstatus}")

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Download
    child = pexpect.spawn(f"{downloader} get {repo_path} {root_dir}")
    done = False
    while not done:
        try:
            # Decode and print the chunk
            chunk = child.read_nonblocking(size=1024, timeout=1)
            print(chunk.decode("utf-8", errors="replace"), end="")
        except pexpect.TIMEOUT:
            pass
        except pexpect.EOF:
            done = True

    child.close()
    print(f"repocli get exited with status {child.exitstatus}")
    return


class RateLimiter:
    def __init__(self, max_calls=100, period=60):
        """To avoid error 429 (too many requests). Period in seconds"""
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.calls = 0
        self.start_time = time.time()

    def wait(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.start_time
                # Reset counter if period has passed
                if elapsed >= self.period:
                    self.calls = 0
                    self.start_time = now
                    elapsed = 0
                # No wait if rate not exceeded
                if self.calls < self.max_calls:
                    self.calls += 1
                    return
                # Calculate time to wait until the period resets
                wait_time = self.period - elapsed
            time.sleep(wait_time)


def download_osf(root_dir="data/gwilliams2023", project_ids: list[str] = []):
    """Downloads data through the open science framework cli."""

    username = input("Enter OSF username (email address): ")
    password = input("Enter OSF password: ")
    token = input("Enter OSF personal access token: ")

    osf = osfclient.OSF(
        username=username,
        password=password,
        token=token,
    )

    osf.login(username=username, password=password, token=token)

    print(f"Logged in as {username}. Gathering files for {root_dir}.")

    download_tasks = []

    rate_limiter = RateLimiter(max_calls=90, period=60)

    # Even listing storage dir is a bit slow, so parallelize
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # One job per project
        future_to_project = {
            executor.submit(gather_osf_files, osf, root_dir, pid): pid
            for pid in project_ids
        }
        # Collect results
        for future in concurrent.futures.as_completed(future_to_project):
            project_id = future_to_project[future]
            try:
                result = future.result()
                download_tasks.extend(result)
            except Exception as exc:
                print(f"Error gathering files for project {project_id}: {exc}")

    total_files = len(download_tasks)
    if total_files == 0:
        print("No files found for given project IDs.")
        return

    print(f"Found {total_files} files. Downloading after rate limit reset (1 min)...")
    time.sleep(60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as executor:
        futures = [
            executor.submit(download_osf_file, file_, path, rate_limiter)
            for (file_, path) in download_tasks
        ]

        with tqdm.tqdm(total=len(futures), unit="files") as pbar:
            for future in concurrent.futures.as_completed(futures):
                _ = future.result()
                pbar.update(1)

    failed = 0
    for file, path in download_tasks:
        if not os.path.exists(path):
            print(f"Failed to download {file.path} to {path}")
            failed += 1
    print(f"Downloaded {total_files - failed}/{total_files} files. Returning...")

    return


def gather_osf_files(osf, root_dir: str, project_id: str):
    """
    Worker function to list all files from a single osf project,
    returning a list of (file_obj, local_path) tuples.
    """
    project = osf.project(project_id)
    download_tasks = []

    for store in project.storages:
        for file_ in store.files:
            # Clean up make target path
            path = file_.path.lstrip("/")
            path = os.path.join(root_dir, path)

            # Create directories
            directory, _ = os.path.split(path)
            os.makedirs(directory, exist_ok=True)

            # Only download if the file does not exist
            if not os.path.exists(path):
                download_tasks.append((file_, path))
    return download_tasks


def download_osf_file(file_, target_path: str, rate_limiter: RateLimiter):
    """
    Worker function to download a single file from OSF to local path
    without displaying any tqdm progress bar.
    """
    import osfclient.models.file

    # no-tqdm copy function for osfclient
    def copyfileobj_no_tqdm(fsrc, fdst, total, length=16 * 1024):
        """Copy data from file-like object fsrc to file-like object fdst
        but WITHOUT a progress bar.
        """
        while True:
            buf = fsrc.read(length)
            if not buf:
                break
            fdst.write(buf)

    # swap out the original copyfileobj
    old_copyfileobj = osfclient.models.file.copyfileobj
    osfclient.models.file.copyfileobj = copyfileobj_no_tqdm

    try:
        # download
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        rate_limiter.wait()
        with open(target_path, "wb") as f:
            file_.write_to(f)
    finally:
        # restore
        osfclient.models.file.copyfileobj = old_copyfileobj

    return


def compute_sha256(filepath):
    """Compute the SHA-256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        return None
    except PermissionError:
        print(f"PERMISSION DENIED: {filepath}")
        return None


def verify_manifest(manifest_path, root_dir="."):
    """
    Verify the integrity of files listed in a manifest.

    :param manifest_path: Path to the manifest file.
    :param root_dir: Root directory where files are located.
    """
    if not os.path.isfile(manifest_path):
        print(f"Manifest file not found: {manifest_path}")
        return

    with open(manifest_path, "r") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            line = line.strip()
            if not line or line.startswith("#"):  # Skip empty lines and comments
                continue

            parts = line.split(" ", 1)  # Split into two parts: hash and filename
            if len(parts) != 2:
                print(f"INVALID FORMAT on line {line_number}: {line}")
                continue

            recorded_hash, filename = parts
            # Remove any leading/trailing whitespace from filename
            filename = filename.strip()

            # Construct the full path to the file
            file_path = os.path.join(root_dir, filename)

            # Compute the actual checksum
            actual_hash = compute_sha256(file_path)

            if actual_hash is None:
                # File is missing or unreadable
                if "emptyroom" not in filename:
                    print(f"MISSING: {filename}")
            elif actual_hash.lower() != recorded_hash.lower():
                # Hashes do not match
                if "emptyroom" not in filename:
                    print(f"FAILED:   {filename}")
    return
