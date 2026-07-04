from pathlib import Path
import json
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

class OKFGenerator:

    TYPE_FOLDER_MAP = {
        "Disease": "diseases",
        "Symptom": "symptoms",
        "SevereSymptom": "severe_symptoms",
        "Pathogen": "pathogens",
        "Parasite": "parasites",
        "Virus": "viruses",
        "Bacteria": "bacteria",
        "Species": "species",
        "Vector": "vectors",
        "Transmission": "transmissions",
        "Medicine": "medicines",
        "Treatment": "treatments",
        "Vaccine": "vaccines",
        "DiagnosticTest": "diagnostic_tests",
        "Complication": "complications",
        "Prevention": "prevention",
        "RiskFactor": "risk_factors",
        "Population": "populations",
        "Country": "countries",
        "Region": "regions",
        "Organization": "organizations",
        "WHOProgram": "who_programs",
        "Policy": "policies",
        "Statistic": "statistics",
        "Technology": "technologies"
    }

    def __init__(self):

        self.root = Path(__file__).resolve().parent.parent

        self.okg_dir = self.root / "okg"

        self.okf_dir = self.root / "okf"

        self.nodes = {}

        self.edges = []

        self.incoming = defaultdict(list)

        self.outgoing = defaultdict(list)

        self.documents = []

    def load_json_files(self):

        json_files = sorted(self.okg_dir.glob("*.json"))

        if not json_files:
            raise FileNotFoundError("No JSON files found inside okg/")

        print(f"\nFound {len(json_files)} JSON files.\n")

        for file in json_files:

            print(f"Loading {file.name}")

            with open(file, "r", encoding="utf-8") as f:
                graph = json.load(f)

            self.documents.append(graph)

        print("\nFinished loading all JSON files.\n")

    def merge_nodes(self):

        print("Merging nodes...")

        for graph in self.documents:

            for node in graph.get("nodes", []):

                node_id = node["id"]

                if node_id not in self.nodes:

                    self.nodes[node_id] = deepcopy(node)

                    continue

                existing = self.nodes[node_id]

                existing_aliases = set(existing.get("aliases", []))

                new_aliases = set(node.get("aliases", []))

                existing["aliases"] = sorted(
                    existing_aliases | new_aliases
                )

                existing_docs = set(
                    existing.get("source_documents", [])
                )

                new_docs = set(
                    node.get("source_documents", [])
                )

                existing["source_documents"] = sorted(
                    existing_docs | new_docs
                )

                refs = existing.get("references", [])

                for ref in node.get("references", []):

                    if ref not in refs:
                        refs.append(ref)

                existing["references"] = refs

                if not existing.get("description") and node.get("description"):

                    existing["description"] = node["description"]

                existing_properties = existing.get("properties", {})

                for key, value in node.get("properties", {}).items():

                    if key not in existing_properties:

                        existing_properties[key] = value

                existing["properties"] = existing_properties

        print(f"Unique nodes : {len(self.nodes)}")

    def collect_edges(self):

        print("\nCollecting relationships...")

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

                    refs = edge_map[key].get("references", [])

                    for ref in edge.get("references", []):

                        if ref not in refs:
                            refs.append(ref)

                    edge_map[key]["references"] = refs

        self.edges = list(edge_map.values())

        print(f"Unique edges : {len(self.edges)}")

    def build_relationship_maps(self):

        print("\nBuilding relationship maps...")

        for edge in self.edges:

            self.outgoing[
                edge["source"]
            ].append(edge)

            self.incoming[
                edge["target"]
            ].append(edge)

        print("Relationship maps created.")

    def folder_for_type(self, node_type):

        folder = self.TYPE_FOLDER_MAP.get(node_type)

        if folder is None:
            raise ValueError(f"Unknown node type : {node_type}")

        path = self.okf_dir / folder

        path.mkdir(
            parents=True,
            exist_ok=True
        )

        return path

    def markdown_path(self, node):

        return self.folder_for_type(
            node["type"]
        ) / f"{node['id']}.md"
    
    def format_properties(self, properties):

        if not properties:
            return "None"

        lines = []

        for key, value in properties.items():
            lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def format_aliases(self, aliases):

        if not aliases:
            return "None"

        return ", ".join(sorted(aliases))

    def format_source_documents(self, documents):

        if not documents:
            return "None"

        lines = []

        for doc in sorted(documents):
            lines.append(f"- {doc}")

        return "\n".join(lines)

    def format_references(self, references):

        if not references:
            return "None"

        lines = []

        seen = set()

        for ref in references:

            key = (
                ref.get("source", ""),
                ref.get("url", "")
            )

            if key in seen:
                continue

            seen.add(key)

            source = ref.get("source", "Unknown")

            url = ref.get("url", "")

            lines.append(f"- **{source}**: {url}")

        return "\n".join(lines)

    def format_outgoing(self, node_id):

        edges = self.outgoing.get(node_id, [])

        if not edges:
            return "None"

        lines = []

        for edge in sorted(
            edges,
            key=lambda e: (e["relation"], e["target"])
        ):

            lines.append(
                f"- **{edge['relation']}** → [[{edge['target']}]]"
            )

        return "\n".join(lines)

    def format_incoming(self, node_id):

        edges = self.incoming.get(node_id, [])

        if not edges:
            return "None"

        lines = []

        for edge in sorted(
            edges,
            key=lambda e: (e["relation"], e["source"])
        ):

            lines.append(
                f"- [[{edge['source']}]] → **{edge['relation']}**"
            )

        return "\n".join(lines)

    def generate_markdown(self, node):

        markdown = f"""# {node['label']}

## ID

{node['id']}

## Type

{node['type']}

## Aliases

{self.format_aliases(node.get('aliases', []))}

## Description

{node.get('description', '') or 'None'}

## Properties

{self.format_properties(node.get('properties', {}))}

## Source Documents

{self.format_source_documents(node.get('source_documents', []))}

## Outgoing Relationships

{self.format_outgoing(node['id'])}

## Incoming Relationships

{self.format_incoming(node['id'])}

## References

{self.format_references(node.get('references', []))}
"""

        return markdown

    def write_readme(self):

        print("\nGenerating README.md")

        readme = []

        readme.append("# WHO Disease Knowledge Repository\n")

        readme.append(
            "This repository was automatically generated from WHO Disease Reports using the Open Knowledge Graph (OKG) pipeline.\n"
        )

        readme.append("## Repository Statistics\n")

        readme.append(f"- **Documents:** {len(self.documents)}")

        readme.append(f"- **Nodes:** {len(self.nodes)}")

        readme.append(f"- **Relationships:** {len(self.edges)}")

        readme.append(
            f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

        readme.append("## Folder Structure\n")

        for node_type in sorted(self.TYPE_FOLDER_MAP.keys()):

            folder = self.TYPE_FOLDER_MAP[node_type]

            count = sum(
                1
                for node in self.nodes.values()
                if node["type"] == node_type
            )

            readme.append(
                f"- **{folder}/** ({count} files)"
            )

        readme.append("\n## Main Files\n")

        readme.append("- index.md")

        readme.append("- Generated automatically by generate_okf.py")

        readme.append(
            "- Source: WHO Disease Reports"
        )

        readme_path = self.okf_dir / "README.md"

        with open(readme_path, "w", encoding="utf-8") as f:

            f.write("\n".join(readme))

        print("Created okf/README.md")

        readme.append("\n## Ontology\n")

        readme.append("| Node Type | Folder |")

        readme.append("|-----------|--------|")

        for node_type, folder in sorted(self.TYPE_FOLDER_MAP.items()):

            readme.append(
                f"| {node_type} | {folder} |"
            )

    def build_index(self):

        index = []

        index.append("# WHO Disease Knowledge Graph\n")

        index.append("Automatically generated from WHO Disease Reports.\n")

        total_nodes = len(self.nodes)

        total_edges = len(self.edges)

        index.append(f"**Total Nodes:** {total_nodes}")

        index.append(f"**Total Relationships:** {total_edges}\n")

        grouped = defaultdict(list)

        for node in self.nodes.values():
            grouped[node["type"]].append(node)

        for node_type in sorted(grouped.keys()):

            nodes = sorted(
                grouped[node_type],
                key=lambda n: n["label"]
            )

            folder = self.TYPE_FOLDER_MAP[node_type]

            index.append(
                f"\n## {folder.replace('_',' ').title()} ({len(nodes)})\n"
            )

            for node in nodes:

                index.append(
                    f"- [[{node['id']}]]"
                )

        return "\n".join(index)
    
    def write_markdown_files(self):

        print("\nGenerating markdown files...\n")

        count = 0

        for node in sorted(
            self.nodes.values(),
            key=lambda n: (n["type"], n["label"])
        ):

            path = self.markdown_path(node)

            markdown = self.generate_markdown(node)

            path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(path, "w", encoding="utf-8") as f:
                f.write(markdown)

            count += 1

            print(f"Created {path.relative_to(self.root)}")

        print(f"\nGenerated {count} markdown files.")

    def write_index(self):

        print("\nGenerating index.md")

        index_path = self.okf_dir / "index.md"

        with open(index_path, "w", encoding="utf-8") as f:

            f.write(self.build_index())

        print("Created okf/index.md")

    def validate_graph(self):

        print("\nValidating graph...")

        errors = []

        for edge in self.edges:

            if edge["source"] not in self.nodes:

                errors.append(
                    f"Missing source node: {edge['source']}"
                )

            if edge["target"] not in self.nodes:

                errors.append(
                    f"Missing target node: {edge['target']}"
                )

        if errors:

            print("\nValidation errors found:\n")

            for err in errors:
                print(err)

            raise Exception("Graph validation failed.")

        print("Graph validation passed.")

    def print_summary(self):

        print("\n----------------------------------------")

        print("Knowledge Graph Summary")

        print("----------------------------------------")

        print(f"Documents : {len(self.documents)}")

        print(f"Nodes     : {len(self.nodes)}")

        print(f"Edges     : {len(self.edges)}")

        print("----------------------------------------\n")

    def run(self):

        print("\n========================================")

        print("WHO OKG → OKF Generator")

        print("========================================")

        self.load_json_files()

        self.merge_nodes()

        self.collect_edges()

        self.build_relationship_maps()

        self.validate_graph()

        self.write_markdown_files()

        self.write_index()

        self.write_readme()

        self.print_summary()

        print("Finished.\n")


def main():

    generator = OKFGenerator()

    generator.run()


if __name__ == "__main__":

    main()