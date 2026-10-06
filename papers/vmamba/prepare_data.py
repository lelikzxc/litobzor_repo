"""Fetch the pinned author Subset A archive, or verify and copy a local archive."""

import argparse
import hashlib
import urllib.request
from pathlib import Path

REVISION = "c2f14503979b32ac7b4f0bd1a128ddec1eaaf6ef"
SHA256 = "28ed870da08261ecf305b8459f97d36d91579e065eb4179718779a0d5f43fb88"
URL = f"https://raw.githubusercontent.com/yijiazhang666/VMamba-for-semiconductor/{REVISION}/WM811k_Dataset.zip"


def prepare(output, archive=None):
    output = Path(output)
    payload = (
        Path(archive).read_bytes() if archive else urllib.request.urlopen(URL, timeout=60).read()
    )
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise ValueError("Author archive checksum mismatch")
    if output.exists() and output.read_bytes() != payload:
        raise ValueError(f"Refusing to overwrite a different dataset at {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    print(f"Verified author Subset A: {output} (902 images, SHA256={SHA256})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="datasets/vmamba_author/WM811k_Dataset.zip")
    parser.add_argument(
        "--archive", default=None, help="Verify a previously downloaded local archive"
    )
    args = parser.parse_args()
    prepare(args.output, args.archive)
