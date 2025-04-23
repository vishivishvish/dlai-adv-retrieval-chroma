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

- 

***WIP - More Notes Incoming!***
