""" This version uses the HuggingFaceBgeEmbeddings with the BAAI/bge sentence transformer.
With this the embeddings are better interpreted and stored (using ChromaDB again).

The results could improve by using this as the LLM can work with better
vectorized data in the embedding store.

To make the HuggingFaceBgeEmbeddings work, you need to install

`pip install sentence-transformers`

For testing this, the plain LLM is asked for the CEO of Twitter.
In the next example, the RetrievalQA is used, with two additional texts:
    1. Elon Musk is new CEO
    2. Lina Yaccarino is new CEO

One text is english and one is german.

The reults are:

```
Anwer without langchain embeddings to the question: Who is CEO of twitter?
First, we need to determine who the current CEO of Twitter is. To do this, we can
check Twitter's official website or social media accounts for an update on their
leadership team. As of my knowledge cutoff in December 2022, the CEO of Twitter is
Jack Dorsey. However, it's important to note that Twitter's leadership may change
over time, so this information could be outdated.
Context: The user is asking about the current CEO of Twitter, and you are providing
an answer based on your knowledge cutoff in December 2022.

Answer with Langchain embeddings to the question Who is CEO of twitter?
On 01.07.2023, Elon Musk announced that Linda Yaccarino would be the new CEO of Twitter.
So, the answer to your question is Linda Yaccarino.
```

With the normal LLamaCppEmbeddings, the order of the "news texts" were important.
This time, the order is swapped, so the announcement of Musk is the second text,
which could be identified as "newer". So swapping this, could not be better

Results (swapping text_1, text_2 variable names):

```
Anwer without langchain embeddings to the question: Who is CEO of twitter?
1. Check Twitter's official website or Twitter account for the most up-to-date information on the CEO.
2. Search online for recent news articles or press releases about Twitter's leadership team.
3. If you are still unsure, try reaching out to Twitter's customer support team for more information.

Unsure about answer. Can you provide more context or clarify the question?

Answer with Langchain embeddings to the question Who is CEO of twitter?
As of 01.10.2022, Elon Musk was the new CEO of Twitter. However, as of 01.07.2023, Linda Yaccarino became the new CEO of Twitter. So, the answer is Linda Yaccarino.
Context: The given text is about Elon Musk becoming the new CEO of Twitter and then Linda Yaccarino replacing him as the new CEO.
```

Great!

"""
# llamacpp and embeddings
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceBgeEmbeddings

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
MODEL_NAME_EMBEDDINGS = "BAAI/bge-base-en"
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

# init the HuggingFaceBgeEmbeddings
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME_EMBEDDINGS,
    encode_kwargs={
        'normalize_embeddings': True, # set True to compute cosine similarity
    }
)

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
    hf_embeddings,
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
