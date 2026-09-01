"""Rebuild the v36 dataset tree inventory after the conversion-report approval promotion."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import relocate_v35_dataset as relocate

root = Path("/iris/u/kewalk/memory_project/data/lerobot/yam/bin_memory_0830_0831_v36_subtask")
scanned = relocate.build_inventory(root, enforce_v35_identity=False)
inventory = relocate.TreeInventory(
    dataset_repo_id="yam/bin_memory_0830_0831_v36_subtask",
    directories=scanned.directories,
    files=scanned.files,
)
out = Path("/iris/u/kewalk/memory_project/data/0830_0831_v36_dataset_tree_inventory.json")
tmp = out.with_suffix(".json.tmp")
tmp.write_bytes(inventory.canonical_bytes())
tmp.replace(out)
import hashlib
print("inventory sha256:", hashlib.sha256(out.read_bytes()).hexdigest())
print("tree_sha256:", inventory.tree_sha256)
print("file_count:", len(inventory.files), "total_bytes:", inventory.total_bytes)
