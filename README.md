# WHO Knowledge Graph RAG

An ontology-driven Knowledge Graph Retrieval-Augmented Generation (KG-RAG) system built from WHO disease reports.

The project automatically converts WHO fact sheets into an Open Knowledge Graph (OKG), validates the graph, generates an Open Knowledge Format (OKF) repository, builds semantic vector embeddings, and provides a local RAG chatbot powered by Ollama.

---

# Features

- Extracts structured knowledge from WHO disease reports
- Builds an Open Knowledge Graph (OKG)
- Validates ontology and graph consistency
- Generates a human-readable Open Knowledge Format (OKF) repository
- Creates semantic node embeddings directly from the knowledge graph
- Stores embeddings in a FAISS vector database
- Retrieves relevant knowledge graph nodes
- Answers questions using a local Large Language Model (Qwen3 via Ollama)
- Fully offline after knowledge graph extraction

---

# Project Architecture

```
WHO PDF Reports
        │
        ▼
Gemini Extraction
        │
        ▼
Open Knowledge Graph (JSON)
        │
        ├──────────────► OKF Repository (Markdown)
        │
        └──────────────► Semantic Node Documents
                                │
                                ▼
                          FAISS Vector Database
                                │
                                ▼
                          Knowledge Retriever
                                │
                                ▼
                        Ollama (Qwen3:8B)
                                │
                                ▼
                           Final Answer
```

---

# Project Structure

```
RAG/
│
├── okg/
│   ├── dengue.json
│   ├── malaria.json
│   └── tuberculosis.json
│
├── okf/(Need to be generated using generate_okf.py)
│   ├── diseases/
│   ├── symptoms/
│   ├── medicines/
│   ├── vectors/
│   ├── countries/
│   ├── ...
│   ├── README.md
│   └── index.md
│
├── prompts/
│   └── extraction_prompt.txt
│
├── schema/
│   └── okg_schema_v1.json
│
├── scripts/
│   ├── validate_schema.py
│   ├── validate_okg.py
│   ├── generate_okf.py
│   └── create_nodes.py
│
├── rag/
│   ├── build_vectors.py
│   ├── retrieve.py
│   └── chatbot.py
│
├── vectordb/
│   ├── okg.index
│   ├── documents.pkl
│   └── metadata.pkl
│
└── README.md
```

---

# Open Knowledge Graph (OKG)

The `okg/` directory contains the structured knowledge graph extracted from WHO disease reports.

Each JSON file contains:

- Nodes
- Relationships
- Properties
- References
- Source documents

Example

```json
{
    "nodes": [...],
    "edges": [...]
}
```

The JSON files represent the canonical knowledge graph used by the RAG system.

---

# Open Knowledge Format (OKF)

The `okf/` directory is a human-readable representation of the knowledge graph.

Each entity becomes a Markdown document.

Example

```
okf/

diseases/
    malaria.md

symptoms/
    fever.md

vectors/
    aedes_aegypti.md
```

Each Markdown file contains

- Entity information
- Description
- Properties
- Relationships
- References
- Source documents

The OKF repository is generated automatically from the knowledge graph.

The chatbot does **not** use these Markdown files during retrieval.

---

# scripts/

Contains utilities for validating and generating the knowledge repository.

## validate_schema.py

Validates every JSON file against the ontology schema.

Checks

- Required fields
- Missing properties
- Invalid node types
- Invalid edge types

Run

```bash
python scripts/validate_schema.py
```

---

## validate_okg.py

Validates the knowledge graph itself.

Checks

- Duplicate node IDs
- Broken relationships
- Duplicate labels
- Alias collisions
- Ontology consistency

Run

```bash
python scripts/validate_okg.py
```

---

## generate_okf.py

Generates the complete Markdown knowledge repository.

Creates

- README.md
- index.md
- Markdown node files

Run

```bash
python scripts/generate_okf.py
```

---

# rag/

Contains the Retrieval-Augmented Generation pipeline.

## build_vectors.py

Reads the Open Knowledge Graph.

Creates one semantic document per node.

Example

```
Node: Malaria

Description:
...

Relationships

Malaria has symptom Fever.

Malaria is treated by ACT.

...
```

These semantic documents are embedded using

```
BAAI/bge-large-en-v1.5
```

Embeddings are stored in FAISS.

Run

```bash
python rag/build_vectors.py
```

---

## retrieve.py

Loads

- FAISS index
- Semantic documents
- Metadata

Performs semantic retrieval over knowledge graph nodes.

Example

```bash
python rag/retrieve.py
```

Example question

```
Which diseases are mosquito borne?
```

---

## chatbot.py

Combines

- FAISS retrieval
- Semantic context construction
- Ollama
- Qwen3

Produces grounded answers using only WHO knowledge.

Run

```bash
python rag/chatbot.py
```

---

# vectordb/

Generated automatically.

Contains

```
okg.index
```

FAISS vector index.

```
documents.pkl
```

Semantic documents for every knowledge graph node.

```
metadata.pkl
```

Metadata including

- node_id
- title
- node_type
- references
- aliases
- source documents

---

# Workflow

## Step 1

Extract WHO report into JSON

```
WHO PDF

↓

Gemini

↓

OKG JSON
```

Store JSON inside

```
okg/
```

---

## Step 2

Validate ontology

```bash
python scripts/validate_schema.py
```

---

## Step 3

Validate graph

```bash
python scripts/validate_okg.py
```

---

## Step 4

Generate Markdown repository

```bash
python scripts/generate_okf.py
```

---

## Step 5

Build semantic vector database

```bash
python rag/build_vectors.py
```

---

## Step 6

Test retrieval

```bash
python rag/retrieve.py
```

---

## Step 7

Start chatbot

```bash
python rag/chatbot.py
```

---

# Technology Stack

Knowledge Graph

- Open Knowledge Graph (OKG)
- Open Knowledge Format (OKF)

Embeddings

- BAAI/bge-large-en-v1.5

Vector Database

- FAISS

LLM

- Ollama
- Qwen3:8B

Programming Language

- Python

Knowledge Source

- WHO Disease Fact Sheets

---

# Current Knowledge Base

Currently includes

- Dengue
- Malaria
- Tuberculosis

The system is designed to scale by simply adding more WHO disease JSON files to the `okg/` directory and rerunning the validation, OKF generation, and vector-building scripts.

---

# Future Work

- Neo4j graph database integration
- Graph traversal during retrieval
- Hybrid retrieval (Graph + Vector)
- Cross-encoder reranking
- Web interface using Streamlit
- Automatic ontology enrichment
- Multi-document reasoning
- Multi-hop graph retrieval

---

# License

This project is intended for research and educational purposes.

Knowledge is extracted from publicly available WHO disease reports.
