from pathlib import Path
import json
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent.parent

SCHEMA_PATH = ROOT / "schema" / "okg_schema_v1.json"

OKG_DIR = ROOT / "okg"

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

validator = Draft202012Validator(schema)

print("\nSchema Validation\n")

success = True

for json_file in sorted(OKG_DIR.glob("*.json")):

    print(f"Checking {json_file.name}")

    with open(json_file, "r", encoding="utf-8") as f:
        graph = json.load(f)

    errors = sorted(
        validator.iter_errors(graph),
        key=lambda e: e.path
    )

    if not errors:

        print("✓ Valid\n")

        continue

    success = False

    print("Errors:\n")

    for err in errors:

        location = ".".join(str(x) for x in err.path)

        print(f"{location}")

        print(err.message)

        print()

if success:

    print("--------------------------------")

    print("All JSON files conform to schema.")

    print("--------------------------------")

else:

    print("--------------------------------")

    print("Schema validation failed.")

    print("--------------------------------")