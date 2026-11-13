"""Split large traces by condition and verify lossless publication packaging."""
import argparse
from contextlib import ExitStack
from hashlib import sha256
import gzip
import json
from pathlib import Path


def package(folder):
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source = folder / "requests.csv.gz"
    if not source.exists():
        raise ValueError("No unsplit trace found")
    target = folder / "traces"
    target.mkdir(exist_ok=False)
    counts = {}
    digest = sha256()
    with gzip.open(source, "rt", newline="") as stream, ExitStack() as stack:
        header = stream.readline()
        assert header.rstrip().split(",")[1] == "condition"
        digest.update(header.encode())
        outputs = {}
        for condition in sorted(manifest["conditions"]):
            outputs[condition] = stack.enter_context(gzip.open(target / f"requests_{condition}.csv.gz", "wt", newline=""))
            outputs[condition].write(header)
            counts[condition] = 0
        for line in stream:
            condition = line.split(",", 2)[1]
            outputs[condition].write(line)
            counts[condition] += 1
            digest.update(line.encode())
    check = sha256()
    parts = []
    for i, condition in enumerate(sorted(counts)):
        path = target / f"requests_{condition}.csv.gz"
        with gzip.open(path, "rt", newline="") as stream:
            first = stream.readline()
            assert first == header
            if i == 0: check.update(first.encode())
            for line in stream: check.update(line.encode())
        parts.append({"path": str(path.relative_to(folder)), "records": counts[condition],
                      "sha256": sha256(path.read_bytes()).hexdigest()})
    assert check.hexdigest() == digest.hexdigest()
    assert sum(counts.values()) == manifest["request_policy_records"]
    manifest["trace_parts"] = parts
    manifest["logical_trace_sha256"] = digest.hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    source.unlink()
    return {"records": sum(counts.values()), "parts": len(parts), "lossless_sha256": digest.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    print(json.dumps(package(parser.parse_args().folder), indent=2))
