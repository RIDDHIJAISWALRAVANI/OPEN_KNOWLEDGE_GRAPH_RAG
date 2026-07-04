from pathlib import Path
import pickle
import re
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent

OKF_DIR = ROOT / "okf"

VECTOR_DIR = ROOT / "vectordb"

VECTOR_DIR.mkdir(exist_ok=True)

MODEL_NAME = "BAAI/bge-large-en-v1.5"

print(f"Loading embedding model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

documents = []

metadata = []


def extract_section(text, section):

    pattern = rf"## {re.escape(section)}\s*(.*?)(?=\n## |\Z)"

    match = re.search(
        pattern,
        text,
        re.DOTALL | re.IGNORECASE
    )

    if not match:
        return ""

    return match.group(1).strip()


def create_semantic_chunks(md_file):

    text = md_file.read_text(encoding="utf-8")

    title_match = re.search(
        r"# (.+)",
        text
    )

    title = (
        title_match.group(1).strip()
        if title_match
        else md_file.stem.replace("_", " ").title()
    )

    type_match = re.search(
        r"## Type\s*\n\s*(.+)",
        text
    )

    node_type = (
        type_match.group(1).strip()
        if type_match
        else md_file.parent.name
    )

    description = extract_section(
        text,
        "Description"
    )

    properties = extract_section(
        text,
        "Properties"
    )

    outgoing = extract_section(
        text,
        "Outgoing Relationships"
    )

    incoming = extract_section(
        text,
        "Incoming Relationships"
    )

    aliases = extract_section(
        text,
        "Aliases"
    )

    chunks = []

    if description:

        chunks.append({
            "section": "Description",
            "text":
                f"{title} is a {node_type}.\n\n"
                f"{description}"
        })

    if properties and properties != "None":

        chunks.append({
            "section": "Properties",
            "text":
                f"{title} ({node_type}) has the following properties:\n\n"
                f"{properties}"
        })

    if outgoing and outgoing != "None":

        chunks.append({
            "section": "Outgoing Relationships",
            "text":
                f"{title} ({node_type}) has these relationships:\n\n"
                f"{outgoing}"
        })

    if incoming and incoming != "None":

        chunks.append({
            "section": "Incoming Relationships",
            "text":
                f"Other entities connected to {title} ({node_type}):\n\n"
                f"{incoming}"
        })

    if aliases and aliases != "None":

        chunks.append({
            "section": "Aliases",
            "text":
                f"{title} is also known as:\n\n"
                f"{aliases}"
        })

    return node_type, title, chunks


print("\nReading markdown files...\n")

markdown_files = sorted(
    OKF_DIR.rglob("*.md")
)

for md_file in tqdm(markdown_files):

    if md_file.name in [
        "README.md",
        "index.md"
    ]:
        continue

    node_type, title, chunks = create_semantic_chunks(md_file)

    for i, chunk in enumerate(chunks):

        documents.append(chunk["text"])

        metadata.append({

            "file":
                str(md_file.relative_to(ROOT)),

            "folder":
                md_file.parent.name,

            "node_id":
                md_file.stem,

            "title":
                title,

            "node_type":
                node_type,

            "section":
                chunk["section"],

            "chunk":
                i,

            "chunk_id":
                f"{md_file.stem}_{i}",

            "chunk_length":
                len(chunk["text"])
        })

print()

print(f"Markdown files : {len(markdown_files)-2}")

print(f"Semantic chunks : {len(documents)}")

print("\nGenerating embeddings...\n")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True,
    convert_to_numpy=True
)

dimension = embeddings.shape[1]

print(f"\nEmbedding dimension : {dimension}")

index = faiss.IndexFlatIP(dimension)

index.add(
    embeddings.astype(np.float32)
)

print("\nSaving FAISS index...\n")

faiss.write_index(
    index,
    str(VECTOR_DIR / "okf.index")
)

with open(VECTOR_DIR / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

with open(VECTOR_DIR / "metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("==========================================")
print("Embedding generation completed successfully")
print("==========================================")

print(f"Markdown files     : {len(markdown_files)-2}")
print(f"Semantic chunks    : {len(documents)}")
print(f"Embedding size     : {dimension}")
print(f"Vectors stored     : {index.ntotal}")

print("\nGenerated files")

print(f"✓ {VECTOR_DIR / 'okf.index'}")
print(f"✓ {VECTOR_DIR / 'metadata.pkl'}")
print(f"✓ {VECTOR_DIR / 'documents.pkl'}")

print("\nChunk Statistics")

sections = {}

for item in metadata:

    section = item["section"]

    sections[section] = sections.get(section, 0) + 1

for section in sorted(sections):

    print(f"{section:<25} {sections[section]}")

print("\nNode Type Statistics")

node_types = {}

for item in metadata:

    node_type = item["node_type"]

    node_types[node_type] = node_types.get(node_type, 0) + 1

for node_type in sorted(node_types):

    print(f"{node_type:<25} {node_types[node_type]}")

print("\nDone.")