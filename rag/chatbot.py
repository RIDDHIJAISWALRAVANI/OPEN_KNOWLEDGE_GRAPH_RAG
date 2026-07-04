from pathlib import Path
from ollama import chat

from retrieve import OKGRetriever


class WHOChatbot:

    MODEL = "qwen3:8b"

    def __init__(self):

        print("\nLoading WHO Knowledge Graph Retriever...\n")

        self.retriever = OKGRetriever()

        print("Retriever loaded.")

        print(f"Using Ollama model : {self.MODEL}\n")

    def build_context(self, results):

        context = []

        references = []

        for i, result in enumerate(results, start=1):

            meta = result["metadata"]

            context.append("=" * 80)

            context.append(
                f"DOCUMENT {i}"
            )

            context.append("=" * 80)

            context.append(
                f"Node : {meta['title']}"
            )

            context.append(
                f"Type : {meta['node_type']}"
            )

            aliases = meta.get(
                "aliases",
                []
            )

            if aliases:

                context.append(
                    "Aliases : "
                    + ", ".join(aliases)
                )

            docs = meta.get(
                "file",
                []
            )

            if docs:

                context.append(
                    "Source Documents : "
                    + ", ".join(docs)
                )

            context.append("")

            context.append(
                result["document"]
            )

            context.append("")

            refs = meta.get(
                "references",
                []
            )

            for ref in refs:

                if ref not in references:

                    references.append(ref)

        context_text = "\n".join(context)

        return context_text, references

    def build_prompt(
        self,
        question,
        context
    ):

        prompt = f"""
You are an expert biomedical assistant.

You answer questions ONLY from the supplied WHO
knowledge graph context.

Rules

1. Never invent facts.

2. Never use your own medical knowledge.

3. If the answer is not present in the context,
reply:

"I could not find this information in the WHO
knowledge graph."

4. Give concise but complete answers.

5. When possible, answer using bullet points.

6. Mention disease names exactly as written.

7. Do not mention that you are an AI model.

WHO KNOWLEDGE GRAPH

{context}

USER QUESTION

{question}

FINAL ANSWER
"""

        return prompt

    def retrieve_context(
        self,
        question,
        top_k=5
    ):

        print("\nSearching Knowledge Graph...\n")

        results = self.retriever.search(
            question,
            top_k
        )

        context, references = self.build_context(
            results
        )

        return (
            results,
            context,
            references
        )

    def ask_llm(
        self,
        prompt
    ):

        print("Generating answer...\n")

        response = chat(
            model=self.MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    def print_references(
        self,
        references
    ):

        if not references:
            return

        print("\n")
        print("=" * 80)
        print("WHO REFERENCES")
        print("=" * 80)

        shown = set()

        for ref in references:

            source = ref.get(
                "source",
                ""
            )

            url = ref.get(
                "url",
                ""
            )

            key = (source, url)

            if key in shown:
                continue

            shown.add(key)

            if source:
                print(f"• {source}: {url}")
            else:
                print(f"• {url}")

    def answer(
        self,
        question
    ):

        results, context, references = self.retrieve_context(
            question
        )

        prompt = self.build_prompt(
            question,
            context
        )

        answer = self.ask_llm(
            prompt
        )

        print("\n")
        print("=" * 80)
        print("ANSWER")
        print("=" * 80)
        print()
        print(answer)

        self.print_references(
            references
        )

    def chat(self):

        print("=" * 80)
        print("WHO KNOWLEDGE GRAPH CHATBOT")
        print("=" * 80)
        print()
        print(f"Model : {self.MODEL}")
        print("Type 'exit' to quit.\n")

        while True:

            question = input("Question: ").strip()

            if question.lower() in [
                "exit",
                "quit"
            ]:
                break

            if not question:
                continue

            try:

                self.answer(question)

            except KeyboardInterrupt:

                print("\nInterrupted.\n")

            except Exception as e:

                print("\nError:")
                print(e)

            print("\n")


def main():

    chatbot = WHOChatbot()

    chatbot.chat()


if __name__ == "__main__":

    main()