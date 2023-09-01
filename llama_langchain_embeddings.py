""" This is the first embedding script, that loads simple text documents
into a vector store and enhances the LLM model with new informations.
To run this, chromadb needs to be installed.

This uses standard llama embeddings and chromadb (pip install chromadb)
For testing this, the plain LLM is asked for the CEO of Twitter.
In the next example, the RetrievalQA is used, with two additional texts:
    1. Elon Musk is new CEO
    2. Lina Yaccarino is new CEO

One text is english and one is german.

(Note the texts are in order of the date, if not it is wrong!)

The reults are:

```
Anwer without langchain embeddings to the question: Who is CEO of twitter?
As of March 2023, Twitter's CEO is Jack Dorsey. He co-founded Twitter and has been the CEO twice, first from 2006 to 2008 and again since 2015.
Context: You are looking for information about the CEO of Twitter.

Answer with Langchain embeddings to the question Who is CEO of twitter?
On 01.07.2023, it was announced that Linda Yaccarino would be the new CEO of Twitter. So, the answer is Linda Yaccarino.
```
"""
# llamacpp and embeddings
from langchain.llms import LlamaCpp
from langchain.embeddings import LlamaCppEmbeddings

# text loaded and processor
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# chroma vector store
from langchain.vectorstores import Chroma

# langchain chain stuff
from langchain.chains import RetrievalQA

# simple Prompt generation stuff (no history)
from langchain import PromptTemplate, LLMChain

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"
MODEL_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": 2048, # context size
    "n_gpu_layers": 10, # layers to shift to qpu if possible
}
PATH_CHROMA_DB = 'local-db' # just local folder

# init the LLM
llm = LlamaCpp(**MODEL_PARAMS)

# init the basic prompt
SIMPLE_PROMPT_TEMPLATE = """
<<SYS>>You answer question of the user. Keep the answer short and concise. Respond "Unsure about answer" if not sure about the answer. If there is a context, use it.<<SYS>>
{context}
Question: {question}
Answer: Let's think step by step.
"""
prompt = PromptTemplate(
    template=SIMPLE_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# create an llm chain to have the primpt
llm_chain = LLMChain(prompt=prompt, llm=llm)

# init the llamacpp embeddings
llama_embeddings = LlamaCppEmbeddings(**MODEL_PARAMS)

## add some simple text embeddings inline
text_1 = """01.10.2022 Elon Musk is new CEO of Twitter"""
text_2 = """
01.07.2023: Linda Yaccarino ist neue CEO von Twitter

Bereits gestern gab Musk bekannt, eine Nachfolgerin für den Posten als Twitter-Chef gefunden zu haben.
Nun ist es offiziell: Die Werbemanagerin Linda Yaccarino wird demnächst die Onlineplattform leiten.
Die Werbemanagerin Linda Yaccarino wird neue CEO der Onlineplattform Twitter. Das gab Twitter-Eigentümer
Elon Musk in dem Kurzbotschaftendienst bekannt und bestätigte damit jüngste US-Medienberichte. Die neue
Geschäftsführerin des Kurznachrichtendienstes leitete zuletzt das Anzeigengeschäft beim Medienriesen NBC Universal.
"""

# setup chroma db
chromadb = Chroma.from_texts(
    [
        text_1,
        text_2
    ],
    llama_embeddings,
    metadatas=[
        {"source": "my-own-source-text-1"},
        {"source": "my-own-source-text-2"}
    ],
    persist_directory=PATH_CHROMA_DB,
)
chromadb.persist()

# create the retrieval qa(question/answer) chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=chromadb.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt,
    },
)

question = "Who is CEO of twitter?"
print(f"Anwer without langchain embeddings to the question: {question}")
print(llm_chain.run({"question": question, "context": ""}))
print(f"Answer with Langchain embeddings to the question {question}")
print(qa.run(question))
