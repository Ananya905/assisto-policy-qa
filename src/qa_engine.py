from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(vector_db):

    prompt_template = """
    You are a Policy Document Assistant.

    Rules:
    1. Explain answers in simple language.
    2. Always mention the exact policy section or clause.
    3. If the answer is not found, say "Not covered in the policy".

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
