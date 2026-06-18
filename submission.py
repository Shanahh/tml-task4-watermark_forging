import os
import sys
import requests
from pathlib import Path


BASE_URL = "http://34.63.153.158"
API_KEY  = "YOUR_API_KEY_HERE"  # REPLACE WITH YOUR API KEY
TASK_ID   = "22-forging-task"
FILE_PATH = Path("/PATH/FILE.zip")

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)


if not os.path.isfile(FILE_PATH):
    die(f"File not found: {FILE_PATH}")

try:
    with open(FILE_PATH, "rb") as f:
        files = {
            "file": (os.path.basename(FILE_PATH), f, "zip"),
        }
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files=files,
            timeout=(50, 300),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

    resp.raise_for_status()

    submission_id = body.get("submission_id")
    print("Successfully submitted.")
    print("Server response:", body)
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)
    