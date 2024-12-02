import time
from langchain.chains import RetrievalQA

def load_qa_chain(retriever, llm, prompt):
    print("Loading QA chain...")
    start_time = time.time()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    print(f"QA chain load time: {time_taken} seconds.\n")
    return qa_chain


def get_response(query, chain) -> str:
    response = chain({"query": query})
    res = response["result"]
    return res
