"""Fetch the pinned author Subset A archive, or verify and copy a local archive."""

import argparse
import hashlib
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REVISION = "c2f14503979b32ac7b4f0bd1a128ddec1eaaf6ef"
SHA256 = "28ed870da08261ecf305b8459f97d36d91579e065eb4179718779a0d5f43fb88"
URL = f"https://raw.githubusercontent.com/yijiazhang666/VMamba-for-semiconductor/{REVISION}/WM811k_Dataset.zip"


def archive_sha256(path):
    """Hash the archive without holding a second copy of it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(output, archive=None):
    output = Path(output)
    if output.exists():
        if not output.is_file() or archive_sha256(output) != SHA256:
            raise ValueError(f"Refusing to overwrite a different dataset at {output}")
        # Copying a prepared repository to Windows must also work offline.
        if archive is None:
            print(f"Verified author Subset A: {output} (902 images, SHA256={SHA256})")
            return output

    if archive is not None:
        source = Path(archive)
        if not source.is_file():
            raise FileNotFoundError(f"Local author archive not found: {source}")
        payload = source.read_bytes()
    else:
        try:
            with urllib.request.urlopen(URL, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(
                "Could not download the author WM-811K archive. Copy the verified "
                "datasets/vmamba_author/WM811k_Dataset.zip from the prepared repository, "
                "or retry with --archive <local-path>. "
                f"Download error: {exc}"
            ) from exc
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Author archive checksum mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        temporary = None
        try:
            # Close before replace: Windows disallows renaming an open file.
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=output.parent, prefix=output.name + ".", suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
            temporary.replace(output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    print(f"Verified author Subset A: {output} (902 images, SHA256={SHA256})")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/vmamba_author/WM811k_Dataset.zip")
    parser.add_argument(
        "--archive", default=None, help="Verify a previously downloaded local archive"
    )
    args = parser.parse_args()
    prepare(args.output, args.archive)
