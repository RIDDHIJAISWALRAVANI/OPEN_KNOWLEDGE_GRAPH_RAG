from pathlib import Path
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class OKGRetriever:

    MODEL_NAME = "BAAI/bge-large-en-v1.5"

    def __init__(self):

        self.root = Path(__file__).resolve().parent.parent

        self.vector_dir = self.root / "vectordb"

        print(f"Loading embedding model: {self.MODEL_NAME}")

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

        print("Loading vector database...")

        self.index = faiss.read_index(
            str(self.vector_dir / "okg.index")
        )

        with open(
            self.vector_dir / "documents.pkl",
            "rb"
        ) as f:

            self.documents = pickle.load(f)

        with open(
            self.vector_dir / "metadata.pkl",
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

        print(
            f"Loaded {self.index.ntotal} knowledge graph nodes.\n"
        )

    def embed_query(self, query):

        instruction = (
            "Represent this sentence for retrieving relevant WHO disease knowledge: "
        )

        embedding = self.model.encode(
            [instruction + query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    def search(self, query, top_k=5):

        query_embedding = self.embed_query(query)

        scores, indices = self.index.search(
            query_embedding,
            50
        )

        wanted_types = self.infer_node_types(query)

        results = []

        seen = set()

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.metadata[idx]

            if wanted_types:

                if meta["node_type"] not in wanted_types:
                    continue

            if meta["node_id"] in seen:
                continue

            seen.add(meta["node_id"])

            results.append({

                "score": float(score),

                "metadata": meta,

                "document": self.documents[idx]

            })

            if len(results) == top_k:
                break

        if results:
            return results

        results = []

        seen = set()

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            meta = self.metadata[idx]

            if meta["node_id"] in seen:
                continue

            seen.add(meta["node_id"])

            results.append({
                "score": float(score),
                "metadata": meta,
                "document": self.documents[idx]
            })

            if len(results) == top_k:
                break

        return results
    
    def infer_node_types(self, query):

        query = query.lower()

        mapping = {

            "disease": ["Disease"],
            "diseases": ["Disease"],

            "symptom": ["Symptom", "SevereSymptom"],
            "symptoms": ["Symptom", "SevereSymptom"],

            "medicine": ["Medicine", "Treatment"],
            "medicines": ["Medicine", "Treatment"],

            "treatment": ["Treatment"],
            "treatments": ["Treatment"],

            "vaccine": ["Vaccine"],
            "vaccines": ["Vaccine"],

            "country": ["Country"],
            "countries": ["Country"],

            "region": ["Region"],
            "regions": ["Region"],

            "vector": ["Vector"],
            "vectors": ["Vector"],

            "virus": ["Virus"],
            "viruses": ["Virus"],

            "parasite": ["Parasite"],
            "parasites": ["Parasite"],

            "bacteria": ["Bacteria"],

            "diagnostic": ["DiagnosticTest"],
            "diagnosis": ["DiagnosticTest"],
            "test": ["DiagnosticTest"],
            "tests": ["DiagnosticTest"],

            "prevention": ["Prevention"],
            "prevent": ["Prevention"],

            "risk": ["RiskFactor"],

            "population": ["Population"],
            "populations": ["Population"],

            "organization": ["Organization"],
            "who": ["Organization", "WHOProgram"]
        }

        wanted = set()

        for keyword, node_types in mapping.items():

            if keyword in query:

                wanted.update(node_types)

        return wanted

    def print_results(self, results):

        print()

        print("=" * 90)
        print("WHO KNOWLEDGE GRAPH SEARCH RESULTS")
        print("=" * 90)

        for rank, result in enumerate(results, start=1):

            meta = result["metadata"]

            print()

            print(f"Result #{rank}")

            print("-" * 90)

            print(
                f"Similarity : {result['score']:.4f}"
            )

            print(
                f"Node ID    : {meta['node_id']}"
            )

            print(
                f"Label      : {meta['title']}"
            )

            print(
                f"Type       : {meta['node_type']}"
            )

            aliases = meta.get(
                "aliases",
                []
            )

            if aliases:

                print(
                    "Aliases    : "
                    + ", ".join(aliases)
                )

            docs = meta.get(
                "file",
                []
            )

            if docs:

                print(
                    "Documents  : "
                    + ", ".join(docs)
                )

            refs = meta.get(
                "references",
                []
            )

            if refs:

                print("\nReferences")

                for ref in refs:

                    source = ref.get(
                        "source",
                        ""
                    )

                    url = ref.get(
                        "url",
                        ""
                    )

                    if source:

                        print(
                            f"  • {source}: {url}"
                        )

                    else:

                        print(
                            f"  • {url}"
                        )

            print()

            print("Semantic Document")

            print("-" * 90)

            print(result["document"])

            print()

        print("=" * 90)

    def interactive(self):

        print(
            "\nWHO Knowledge Graph Retriever\n"
        )

        print(
            "Type 'exit' to quit.\n"
        )

        while True:

            query = input(
                "Ask a question: "
            ).strip()

            if query.lower() in [
                "exit",
                "quit"
            ]:
                break

            if not query:

                continue

            print()

            results = self.search(query)

            self.print_results(results)


def main():

    retriever = OKGRetriever()

    retriever.interactive()


if __name__ == "__main__":

    main()