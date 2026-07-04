from pathlib import Path
from collections import defaultdict
from copy import deepcopy
import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class OKGVectorBuilder:

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self):

        self.root = Path(__file__).resolve().parent.parent

        self.okg_dir = self.root / "okg"

        self.vector_dir = self.root / "vectordb"

        self.vector_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(f"Loading embedding model: {self.MODEL_NAME}")

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

        self.documents = []

        self.nodes = {}

        self.edges = []

        self.outgoing = defaultdict(list)

        self.incoming = defaultdict(list)

        self.semantic_documents = []

        self.metadata = []

    def load_json_files(self):

        json_files = sorted(
            self.okg_dir.glob("*.json")
        )

        if not json_files:
            raise FileNotFoundError(
                "No JSON files found in okg/"
            )

        print(f"\nFound {len(json_files)} JSON files.\n")

        for file in json_files:

            print(f"Loading {file.name}")

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                graph = json.load(f)

            self.documents.append(graph)

        print("\nFinished loading JSON files.\n")

    def merge_nodes(self):

        print("Merging nodes...")

        for graph in self.documents:

            for node in graph.get("nodes", []):

                node_id = node["id"]

                if node_id not in self.nodes:

                    self.nodes[node_id] = deepcopy(node)

                    continue

                existing = self.nodes[node_id]

                existing["aliases"] = sorted(
                    set(existing.get("aliases", []))
                    |
                    set(node.get("aliases", []))
                )

                existing["source_documents"] = sorted(
                    set(existing.get("source_documents", []))
                    |
                    set(node.get("source_documents", []))
                )

                refs = existing.get(
                    "references",
                    []
                )

                for ref in node.get(
                    "references",
                    []
                ):

                    if ref not in refs:
                        refs.append(ref)

                existing["references"] = refs

                if (
                    not existing.get("description")
                    and
                    node.get("description")
                ):

                    existing["description"] = node[
                        "description"
                    ]

                props = existing.get(
                    "properties",
                    {}
                )

                for k, v in node.get(
                    "properties",
                    {}
                ).items():

                    if k not in props:
                        props[k] = v

                existing["properties"] = props

        print(
            f"Unique nodes : {len(self.nodes)}"
        )

    def merge_edges(self):

        print("\nMerging edges...")

        edge_map = {}

        for graph in self.documents:

            for edge in graph.get("edges", []):

                key = (
                    edge["source"],
                    edge["relation"],
                    edge["target"]
                )

                if key not in edge_map:

                    edge_map[key] = deepcopy(edge)

                else:

                    refs = edge_map[key].get(
                        "references",
                        []
                    )

                    for ref in edge.get(
                        "references",
                        []
                    ):

                        if ref not in refs:
                            refs.append(ref)

                    edge_map[key][
                        "references"
                    ] = refs

        self.edges = list(
            edge_map.values()
        )

        print(
            f"Unique edges : {len(self.edges)}"
        )

    def build_relationship_maps(self):

        print(
            "\nBuilding relationship maps..."
        )

        for edge in self.edges:

            self.outgoing[
                edge["source"]
            ].append(edge)

            self.incoming[
                edge["target"]
            ].append(edge)

        print(
            "Relationship maps created."
        )

    def node_label(self, node_id):

        node = self.nodes.get(node_id)

        if node is None:
            return node_id

        return node["label"]

    def node_type(self, node_id):

        node = self.nodes.get(node_id)

        if node is None:
            return "Unknown"

        return node["type"]

    RELATION_TEMPLATES = {

        "HAS_SYMPTOM":
            "{source} has symptom {target}.",

        "HAS_SEVERE_SYMPTOM":
            "{source} has severe symptom {target}.",

        "CAUSED_BY":
            "{source} is caused by {target}.",

        "TRANSMITTED_BY":
            "{source} is transmitted by {target}.",

        "TRANSMITTED_VIA":
            "{source} is transmitted via {target}.",

        "TREATED_BY":
            "{source} is treated by {target}.",

        "HAS_VACCINE":
            "{source} has vaccine {target}.",

        "PREVENTED_BY":
            "{source} is prevented by {target}.",

        "DIAGNOSED_BY":
            "{source} is diagnosed by {target}.",

        "AFFECTS":
            "{source} affects {target}.",

        "HIGH_RISK_GROUP":
            "{target} is a high risk group for {source}.",

        "CAUSES":
            "{source} causes {target}.",

        "COMPLICATED_BY":
            "{source} is complicated by {target}.",

        "PREVALENT_IN":
            "{source} is prevalent in {target}.",

        "REPORTED_IN":
            "{source} is reported in {target}.",

        "LOCATED_IN":
            "{source} is located in {target}.",

        "USES":
            "{source} uses {target}.",

        "USES_TECHNOLOGY":
            "{source} uses technology {target}.",

        "RECOMMENDED_BY":
            "{source} is recommended by {target}.",

        "IMPLEMENTED_BY":
            "{source} is implemented by {target}.",

        "TARGETS":
            "{source} targets {target}.",

        "PROTECTS_AGAINST":
            "{source} protects against {target}.",

        "RESISTANT_TO":
            "{source} is resistant to {target}.",

        "RELATED_TO":
            "{source} is related to {target}.",

        "PART_OF":
            "{source} is part of {target}.",

        "MONITORED_BY":
            "{source} is monitored by {target}."
    }

    def relation_to_sentence(self, edge):

        source = self.node_label(edge["source"])

        target = self.node_label(edge["target"])

        relation = edge["relation"]

        template = self.RELATION_TEMPLATES.get(
            relation,
            "{source} " + relation.lower().replace("_", " ") + " {target}."
        )

        return template.format(
            source=source,
            target=target
        )

    def build_semantic_document(self, node):

        lines = []

        lines.append(
            f"Node: {node['label']}"
        )

        lines.append(
            f"Type: {node['type']}"
        )

        if node.get("aliases"):

            aliases = [
                a for a in node["aliases"]
                if a
            ]

            if aliases:

                lines.append("")
                lines.append("Aliases:")

                for alias in aliases:

                    lines.append(
                        f"- {alias}"
                    )

        description = node.get(
            "description",
            ""
        )

        if description:

            lines.append("")
            lines.append("Description:")
            lines.append(description)

        properties = node.get(
            "properties",
            {}
        )

        if properties:

            lines.append("")
            lines.append("Properties:")

            for key, value in properties.items():

                lines.append(
                    f"- {key}: {value}"
                )

        outgoing = self.outgoing.get(
            node["id"],
            []
        )

        if outgoing:

            lines.append("")
            lines.append("Relationships:")

            for edge in outgoing:

                lines.append(
                    self.relation_to_sentence(edge)
                )
        incoming = self.incoming.get(
            node["id"],
            []
        )

        if incoming:

            lines.append("")
            lines.append("Referenced By:")

            for edge in incoming:

                lines.append(
                    self.relation_to_sentence(edge)
                )

        references = node.get(
            "references",
            []
        )

        if references:

            lines.append("")
            lines.append("References:")

            for ref in references:

                source = ref.get("source", "")

                url = ref.get("url", "")

                if source and url:

                    lines.append(
                        f"- {source}: {url}"
                    )

                elif url:

                    lines.append(
                        f"- {url}"
                    )

        return "\n".join(lines)

    def build_semantic_documents(self):

        print("\nBuilding semantic documents...\n")

        for node in self.nodes.values():

            document = self.build_semantic_document(
                node
            )

            self.semantic_documents.append(
                document
            )

            self.metadata.append({

                "node_id":
                    node["id"],

                "title":
                    node["label"],

                "node_type":
                    node["type"],

                "aliases":
                    node.get(
                        "aliases",
                        []
                    ),

                "file":
                    node.get(
                        "source_documents",
                        []
                    ),

                "references":
                    node.get(
                        "references",
                        []
                    )

            })

        print(
            f"Semantic documents : {len(self.semantic_documents)}"
        )

    def preview_documents(
        self,
        limit=3
    ):

        print("\nSample semantic documents\n")

        print("=" * 80)

        for i, document in enumerate(
            self.semantic_documents[:limit],
            start=1
        ):

            print(f"\nDocument {i}\n")

            print(document)

            print("\n" + "=" * 80)

    def generate_embeddings(self):

        print("\nGenerating embeddings...\n")

        embeddings = self.model.encode(
            self.semantic_documents,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        self.embeddings = embeddings.astype(np.float32)

        print(
            f"\nEmbedding dimension : {self.embeddings.shape[1]}"
        )

    def build_faiss_index(self):

        print("\nBuilding FAISS index...\n")

        dimension = self.embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(self.embeddings)

        print(
            f"Vectors indexed : {self.index.ntotal}"
        )

    def save(self):

        print("\nSaving vector database...\n")

        faiss.write_index(
            self.index,
            str(self.vector_dir / "okg.index")
        )

        with open(
            self.vector_dir / "documents.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.semantic_documents,
                f
            )

        with open(
            self.vector_dir / "metadata.pkl",
            "wb"
        ) as f:

            pickle.dump(
                self.metadata,
                f
            )

        print("Saved")

        print(
            f"  {self.vector_dir / 'okg.index'}"
        )

        print(
            f"  {self.vector_dir / 'documents.pkl'}"
        )

        print(
            f"  {self.vector_dir / 'metadata.pkl'}"
        )

    def statistics(self):

        print("\n")
        print("=" * 60)
        print("Knowledge Graph Vector Database")
        print("=" * 60)

        print(
            f"Nodes      : {len(self.nodes)}"
        )

        print(
            f"Edges      : {len(self.edges)}"
        )

        print(
            f"Vectors    : {self.index.ntotal}"
        )

        print(
            f"Dimension  : {self.embeddings.shape[1]}"
        )

        type_counts = defaultdict(int)

        for node in self.nodes.values():

            type_counts[node["type"]] += 1

        print("\nNode Types\n")

        for node_type in sorted(type_counts):

            print(
                f"{node_type:<25} {type_counts[node_type]}"
            )

        print("\n")
        print("=" * 60)

    def run(self):

        self.load_json_files()

        self.merge_nodes()

        self.merge_edges()

        self.build_relationship_maps()

        self.build_semantic_documents()

        self.preview_documents()

        self.generate_embeddings()

        self.build_faiss_index()

        self.save()

        self.statistics()


def main():

    builder = OKGVectorBuilder()

    builder.run()


if __name__ == "__main__":

    main()  