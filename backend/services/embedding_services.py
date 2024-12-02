import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from config import Config


# Function for loading the embedding model
def load_embedding_model(model_name, normalize_embedding=True):
    print("Loading embedding model...")
    start_time = time.time()
    hugging_face_embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE
        },
        encode_kwargs={
            "normalize_embeddings": normalize_embedding
        },
    )
    end_time = time.time()
    time_taken = round(end_time - start_time, 2)
    print(f"Embedding model load time: {time_taken} seconds.\n")
    return hugging_face_embeddings


# Function for creating embeddings using FAISS
def create_embeddings(chunks, embedding_model, storing_path="vectorstore"):
    print("Creating embeddings...")
    e_start_time = time.time()

    vectorstore = FAISS.from_documents(chunks, embedding_model)

    e_end_time = time.time()
    e_time_taken = round(e_end_time - e_start_time, 2)
    print(f"Embeddings creation time: {e_time_taken} seconds.\n")

    print("Writing vectorstore..")
    v_start_time = time.time()

    vectorstore.save_local(storing_path)

    v_end_time = time.time()
    v_time_taken = round(v_end_time - v_start_time, 2)
    print(f"Vectorstore write time: {v_time_taken} seconds.\n")

    return vectorstore
