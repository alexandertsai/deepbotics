"""Store hosted API keys locally without shell history or terminal echo."""

import argparse
import getpass
import json
import os
from pathlib import Path
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--provider", choices=("both", "kimi", "deepseek", "fireworks"), default="both")
args = parser.parse_args()
names = {"kimi": ["MOONSHOT_API_KEY"], "deepseek": ["DEEPSEEK_API_KEY"], "fireworks": ["FIREWORKS_API_KEY"],
         "both": ["MOONSHOT_API_KEY", "DEEPSEEK_API_KEY"]}[args.provider]
path = Path.home() / ".config" / "kimirax" / "api_keys.json"
path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
path.parent.chmod(0o700)
keys = json.loads(path.read_text()) if path.exists() else {}
for name in names:
    value = getpass.getpass(f"{name} (blank keeps existing value): ").strip()
    if value:
        keys[name] = value
fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".api_keys-")
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(keys, stream)
        stream.write("\n")
    os.replace(temp, path)
finally:
    if os.path.exists(temp):
        os.unlink(temp)
print(f"Saved credentials to {path} with permissions 600. Values were not displayed.")
