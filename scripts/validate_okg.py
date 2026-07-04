from pathlib import Path
import json
from collections import defaultdict
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent.parent
OKG_DIR = ROOT / "okg"

nodes = {}
edges = []

label_index = defaultdict(list)
alias_index = defaultdict(list)

print("\nLoading JSON files...\n")

for file in sorted(OKG_DIR.glob("*.json")):

    print(file.name)

    with open(file, "r", encoding="utf-8") as f:
        graph = json.load(f)

    for node in graph.get("nodes", []):

        node_id = node["id"]

        if node_id in nodes:
            print(f"\nDuplicate ID: {node_id}")

        nodes[node_id] = node

        label = node["label"].strip().lower()

        label_index[label].append(node)

        for alias in node.get("aliases", []):

            alias_index[alias.strip().lower()].append(node)

    edges.extend(graph.get("edges", []))

print("\n--------------------------------")

print(f"Nodes : {len(nodes)}")

print(f"Edges : {len(edges)}")

print("--------------------------------")

print("\nChecking node ID consistency...\n")

id_conflicts = False

id_index = defaultdict(list)

# Build index of every node ID across all documents
for file in sorted(OKG_DIR.glob("*.json")):

    with open(file, "r", encoding="utf-8") as f:
        graph = json.load(f)

    for node in graph.get("nodes", []):

        id_index[node["id"]].append(
            {
                "file": file.name,
                "type": node["type"],
                "label": node["label"]
            }
        )

# Check consistency
for node_id, entries in sorted(id_index.items()):

    if len(entries) <= 1:
        continue

    labels = {e["label"] for e in entries}
    types = {e["type"] for e in entries}

    if len(labels) > 1 or len(types) > 1:

        id_conflicts = True

        print(f"\nConflict for node ID: {node_id}")

        for e in entries:

            print(
                f"  {e['file']} | "
                f"{e['type']} | "
                f"{e['label']}"
            )

if not id_conflicts:

    print("✓ Node IDs are globally consistent.")
    

print("\nChecking duplicate labels...\n")

duplicate_found = False

for label, items in label_index.items():

    if len(items) > 1:

        duplicate_found = True

        print(f"\nLabel : {label}")

        for node in items:

            print(f"   {node['id']} ({node['type']})")

print("\nChecking alias collisions...\n")

for alias, items in alias_index.items():

    if len(items) > 1:

        duplicate_found = True

        print(f"\nAlias : {alias}")

        for node in items:

            print(f"   {node['id']} ({node['type']})")

print("\nChecking similar labels...\n")

labels = sorted(label_index.keys())

for i in range(len(labels)):

    for j in range(i + 1, len(labels)):

        ratio = SequenceMatcher(
            None,
            labels[i],
            labels[j]
        ).ratio()

        if ratio > 0.92 and labels[i] != labels[j]:

            print(f"\nPossible duplicate")

            print(labels[i])

            print(labels[j])

print("\nChecking broken edges...\n")

broken = False

for edge in edges:

    if edge["source"] not in nodes:

        broken = True

        print(f"Missing source node : {edge['source']}")

    if edge["target"] not in nodes:

        broken = True

        print(f"Missing target node : {edge['target']}")

if not broken:

    print("✓ No broken edges")

if not duplicate_found:

    print("\n✓ No duplicate labels")

    print("✓ No alias collisions")

print("\nValidation Finished.\n")