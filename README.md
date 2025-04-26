# **Advanced Retrieval for AI with Chroma** 

**[Deeplearning.ai](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)**

**Andrew Ng - CEO, Deeplearning.ai**

**Anton Troynikov - Co-Founder & Advisor, Chroma**

***Notes by Vishnu Subramanian***

## ***1 - Introduction***

- RAG or Retrieval-Augmented Generation retrieves relevant documents to give context to an LLM.
- This makes it much better at answering queries and performing tasks.
- Many teams are using simple Retrieval techniques based on semantic similarity or embeddings, but we learn more sophisticated techniques in this course, which will help us do better than that.
- A common workflow in RAG is to take your query, embed that, then find the most similar documents, meaning the ones with similar embeddings, and that’s the context. 
- But the problem with that is, it can tend to find documents that talk about similar topics as the query, but not actually contain the answer.
- But you can take the initial user query and “Rewrite” - this is called Query Expansion.
- Rewrite the query to pull in more directly related documents.
- There are two key related techniques.
- One, to expand the optional query into multiple queries by rewording or rewriting it in different ways.
- Second, to even guess or hypothesize what the answer might look like to see if we can find anything in our document collection that looks more like an answer rather than only generally talking about the topics of the query.
- The instructor for the course, Anton Troynikov, has been one of the pioneers in driving forward the State-of-the-Art in terms of retrieval for AI applications.
- Anton is the Co-Founder of Chroma, which provides one of the most popular open-source Vector Databases.
- We will start off the course by doing a quick review of RAG applications.
- We will then learn about some of the pitfalls of retrieval, where simple vector search doesn’t do so well, and we’ll learn about some methods to improve the results.
- In the first method, we use an LLM to improve the query itself.
- Another method re-ranks query results, with the help of something called a Cross Encoder, which takes in a pair of sentences and produces a relevancy score.
- We’ll also learn how to adapt the query embeddings based on user feedback to produce more relevant results.
- There’s a lot of innovation going on in RAG right now, so in the final lesson we’ll go over some of the results which aren’t mainstream yet but will become so in the future.

## ***2 - Overview of Embeddings-based Retrieval***

- In the first section, we will review some of the elements of an embedding-based retrieval system, and how that fits together in a RAG loop with an LLM.

<img src="https://drive.google.com/uc?export=view&id=1h7g19LMQYoAdcpPe4sR_a76BnEDMFwHm">

- This is an overall system diagram of how the RAG process works in practice.
- The way RAG works is, you have a user query that comes in, and you have a set of documents which you’ve previously embedded and stored in your retrieval system - in this case, Chroma.
- You take the query, run the query through the same embedding model used to embed your documents, which generates an embedding. 
- The retrieval system then finds the most relevant documents according to the embedding of that query by finding the nearest neighbor embeddings in the database.
- We then provide both the query and the relevant documents to the LLM, and the LLM synthesizes information from the retrieved documents to generate an answer.
- Let’s see how this works in practice.
- To start with, we’re going to pull in some helper functions from our utilities.
- The word_wrap function allows us to look at the documents in a nicely printed way.

`from helper_utils import word_wrap;`

- The example that we’re going to use - we’re going to read from a PDF. So we’ll pull in a PDF Reader from the PyPDF Python package - it’s open-source.

`from pypdf import PdfReader;`

- We’re going to read from Microsoft’s 2022 Annual Report.
- We’re going to extract the texts from the report using this PDF Reader.
- For every page that the reader has, we’re extracting the text and we’re also removing the whitespace.
- Another important thing to make sure of is that we’re not sending any empty pages to the retrieval system - that’s also filtered out.
- In our next step, we need to chunk up these pages, first by character, then by token.
- To do that, we can grab some useful utilities from LangChain.
- We can use the Recursive Character Text Splitter and the Sentence Transformers Token Text Splitter.
- The Character Splitter allows us to divide text recursively according to certain divider characters. The recursive divider characters we’re providing are [‘\n\n’, ‘\n’, ‘. ’, ‘ ’, ‘’], and it will split recursively using these characters targeting a chunk size of 1000.
- But when we run this, we see that 347 chunks have resulted from this document. 
- Character Text Splitting isn’t quite enough, because the Embedding Model we use from Sentence-Transformers, has a limited context window width. In fact, it uses 256 characters.
- That’s the maximum context window length of our embedding model.
- This is a minor pitfall - the embedding model’s context window length is very important, because typically it will simply truncate characters or tokens beyond its context window.
- So to make sure we actually capture all the meaning in each chunk when we embed it, it’s important we also chunk according to the token count keeping in mind the limitations of the embedding model later.
- Hence what we’re doing is we’re using the Sentence Transformer text splitter, with 256 tokens per chunk and overlap of 0.
- We’re going to take all the chunks created by the Character Splitter, and now re-split these chunks using the Sentence Transformer Splitter.

<img src="https://drive.google.com/uc?export=view&id=1jqdzhibSo2k_SDqdEpnFxPHaIZO108t2">

- So now we have our text chunks.
- That’s the first step in any RAG system.
- The next step is to load the chunks that we have into our retrieval.
- In this case, we will be using Chroma, so we’ll import it into our script.

<img src="https://drive.google.com/uc?export=view&id=1-0LlS9CApG57f8s5TJzYkSR4CHakgNQ4">

- In the above example, we have the [CLS] Classifier Token, followed by “I like dogs”.
- In BERT, each token receives its own Dense Vector Embedding.
- A Sentence Transformer, on the other hand, allows us to embed entire sentences or even small documents, by pooling the output of all the Token Embeddings to produce a single dense vector per document, or in our case, per chunk.

<img src="https://drive.google.com/uc?export=view&id=15TMPV1KtE5qx1MY6tB10fgPv4s--TM67">

- Sentence Transformers are great as an Embedding model - they’re open-source, and all the weights are available online. They’re also easy to run locally.
- They come built into Chroma, and we can learn more about them through the Sentence Transformers website.

<img src="https://drive.google.com/uc?export=view&id=1bUgdEy4t5Q9d89CXzl_G_QR1yQftKAMm">

- The above demonstrates what happens when we call the Sentence Transformer Embedding function on the 10th element of the token_split_texts list.
- The output that we get is a very long, dense vector - a 358-dimensional vector that represents the 10th chunk of text.
- We will set up the Chroma DB client, and we create a collection called microsoft_annual_report_2022. 
- We also pass in an Embedding Function as one of the arguments in this collection, which is the Sentence Transformer Embedding Function we defined earlier.

`chroma_client = chromadb.Client()`

`chroma_collection = chroma_client.create_collection("microsoft_annual_report_2022", embedding_function=embedding_function)`

`ids = [str(i) for i in range(len(token_split_texts))]`

`chroma_collection.add(ids=ids, documents=token_split_texts)`

`chroma_collection.count()`

- The next step is to create a list of IDs (the string version of these numbers) to numerically index the embeddings that we’ll store in the Chroma database. We then add the string IDs and the text chunks themselves directly to the Chroma collection.
- The .count() method shows that there are 349 rows loaded into this collection.
- Now, we can connect an LLM and build a full-fledged RAG system.
- We will demonstrate how querying, retrieval and the LLM all work together.

<img src="https://drive.google.com/uc?export=view&id=1m0lDkfibZVeaTdZphUnBfD961ziOsV1b">

- Now, given the query is “What was the total revenue?”, the retrieval is done by sending this query to Chroma DB using the chroma_collection.query() method, and asking for the Top 5 results. 
- From the results, we can retrieve solely the list of chunk texts using the JSON structure of the output by accessing the values inside the ‘documents’ key. 
- Each of the chunk texts inside the list can then be printed using the word_wrap() function to make it more legible, for us to see what we’ve retrieved.
- Now that the relevant chunks have been retrieved, the next step is to use these chunks together with an LLM to answer our query.
- For this, we will load our OpenAI key into the environment so we can authenticate, and we’re going to create an OpenAI client.

`def rag(query, retrieved_documents, model="gpt-3.5-turbo"):`

`    information = "\n\n".join(retrieved_documents)`

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about information contained in an annual report."
            "You will be shown the user's question, and the relevant information from the annual report. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

- In this code block, we are using the “information” variable to append the retrieved_documents into a single string, and inside the prompt to the LLM, we clearly delineate the “query” portion from the “information” portion, to enable the LLM to properly use the contextual information to answer the incoming query, following the guidelines in the System Prompt.
- Now that the rag() function has been defined, putting it all together:

<img src="https://drive.google.com/uc?export=view&id=1wI1PZnPuUXCIeKxCmp6hFTXXCOP-1UgB">

- It’s important to play with the retrieval system to gain intuition about what the model and the retriever can and cannot do together, before we dive into really analyzing how the system works.
- In the next section, we’ll talk about some of the pitfalls and common failure modes of using retrieval in a RAG setting.

## ***3 - Pitfalls of Retrieval - When Simple Vector Search Fails***

- In this section, we will learn about some of the pitfalls of retrieval with vectors.
- We will look at a few examples of where simple vector search is not enough to make retrieval work for our RAG application.
- Just because things are semantically close as vectors in an embedding model, doesn’t always mean you’re going to get good results right out of the box.
- First, we will get set up with our Chroma DB.
- We’re going to use a helper function to load our Chroma collection, and we’re going to load the same Sentence Transformer Embedding Function.

`from helper_utils import load_chroma, word_wrap`

`from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction`

`embedding_function = SentenceTransformerEmbeddingFunction()`

`chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf', collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)`

`chroma_collection.count()`

- 

## ***4 - Query Expansion***

## ***5 - Cross-encoder Re-ranking***

## ***6 - Embedding Adaptors***

## ***7 - Other Techniques***

***WIP - More Notes Incoming!***
