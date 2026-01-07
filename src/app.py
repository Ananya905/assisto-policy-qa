from document_loader import load_and_split_document
from vector_store import create_vector_store
from qa_engine import build_qa_chain

def main():
    print("Loading policy document...")
    chunks = load_and_split_document("data/sample_policy.pdf")

    print("Creating vector database...")
    vector_db = create_vector_store(chunks)

    print("Policy Q&A Assistant is ready 🚀")

    qa_chain = build_qa_chain(vector_db)

    while True:
        question = input("\nAsk a policy question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = qa_chain.run(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
