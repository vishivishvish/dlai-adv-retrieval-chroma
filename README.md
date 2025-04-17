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

- 

***WIP - More Notes Incoming!***
