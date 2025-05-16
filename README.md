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
- The output that we get is a very long, dense vector - a 384-dimensional vector that represents the 10th chunk of text.
- The default Embedding model used by this function is All-MiniLM-L6-v2.
- We will set up the Chroma DB client, and we create a collection called microsoft_annual_report_2022. 
- We also pass in an Embedding Function as one of the arguments in this collection, which is the Sentence Transformer Embedding Function we defined earlier.

```
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection("microsoft_annual_report_2022", embedding_function=embedding_function)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()
```

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

```
def rag(query, retrieved_documents, model="gpt-3.5-turbo"):

    information = "\n\n".join(retrieved_documents)

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
```

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

```
from helper_utils import load_chroma, word_wrap
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_function = SentenceTransformerEmbeddingFunction()

chroma_collection = load_chroma(filename='microsoft_annual_report_2022.pdf', collection_name='microsoft_annual_report_2022', embedding_function=embedding_function)
chroma_collection.count()
```

- We’re also just going to output the count() of the Chroma collection to make sure we’ve got the right number of rows. We’ll see the output of 349, which is what we were expecting.
- When working with embeddings, it can be useful to visualize the Embedding Space.
- Embeddings and their vectors are of course a geometric structure, and you can reason about them spatially.
- Obviously, they’re high-dimensional - 300+ dimensional vectors - but we can project them down into two dimensions which humans can visualize, and this can be useful for reasoning about the structure of the embedding space.
- To do this low-dimensional projection, we’re going to use the Dimensionality Reduction technique called UMAP (Uniform Manifold Approximation & Projection) - an open-source library that can be used for projecting data down into two or three dimensions to visualize it.
- UMAP is similar to techniques like PCA and t-SNE, except, UMAP explicitly tries to preserve the structure of the data in terms of distances between points, as much as it can, unlike PCA for example, which just tries to find the dominant direction and project data down that way.

```
import umap
import numpy as np
from tqdm import tqdm

embeddings = chroma_collection.get(include=['embeddings'])['embeddings']
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
```

- What we’re doing is we’re going to fit a UMAP transform.
- UMAP is basically a model which fits a manifold to your data to project it into two dimensions.
- We set the random seed and the transform seed to 0 so that we get reproducible results and we can get the same projection every time.
- Once we fit the transform, we need to use the transform to project the embeddings.

```
def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings),2))
    for i, embedding in enumerate(tqdm(embeddings)): 
        umap_embeddings[i] = umap_transform.transform([embedding])
    return umap_embeddings  
```

- We define a function called project_embeddings() to do this.
- It takes as input an array of embeddings and a UMAP transform. 
- We start by declaring an empty NumPy array of the same length as our embeddings array, but with dimension 2, because we’re going to get two-dimensional projections out.
- What we’ll do is we’re going to project the embeddings into 2D space one-by-one by running a for loop over them - we do it one-by-one just so that we get consistent behavior from UMAP.
- The way that UMAP does its projection is somewhat sensitive to its inputs, so to ensure that we get reproducible results, we will do it one-by-one, as opposed to doing it in batches, because the output behavior might then differ batch-to-batch.
- Then, we use Matplotlib to do a scatterplot of the projected embeddings.

```
import matplotlib.pyplot as plt
​
plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10)
plt.gca().set_aspect('equal', 'datalim')
plt.title('Projected Embeddings')
plt.axis('off')
```

- We use the plt.scatter() function - we plot the first element from each, the second element from each, and choose size s = 10.

<img src="https://drive.google.com/uc?export=view&id=1ooKtlN123juFNl7LNZszwzUgRchJ8uzj">

- This is what our dataset looks like inside Chroma, projected down to two dimensions.
- A more advanced visualization would allow us to hover over each of these dots, look at the text chunk corresponding to each dot, and we would notice that text chunks with similar meanings would occupy places close to each other even in the 2D projection.
- Sometimes these are unusual structures because a 2D projection cannot represent all the structure of higher dimensional space that the embeddings belong to. But it’s useful for visualization.
- So what evaluating the quality and performance of a retrieval is about, is actually relevancy and distraction.
- Let’s take a look at our original query again - the one we used in our RAG example - “What’s the total revenue?”
- When we examine the 5 chunks retrieved from the Vector DB in response to this query, we see some chunks where the word “revenue” occurs, but the chunk is not actually directly related to answering the question about the total revenue.
- Let’s take a look at how this query looks when visualized.
- We’ll grab the embedding for our query using the embedding function, and we grab the embeddings for the retrieved results of the RAG from the Vector DB.
- We then project both these embeddings into the 2D space created by UMAP - the X represents the query and the circled dots represent the retrieved chunks.

<img src="https://drive.google.com/uc?export=view&id=1fq_aScmb_mAdriIBsDb2uId4pxsP-tqN">

- The heart of the issue is that the Embedding Model that we use to embed our queries and embed our data does not have any knowledge of the task or query we’re trying to answer, at the time we actually retrieve the information. 
- So the reason that a retrieval system may not always perform the way we expect, is that we’re asking it to perform a specific task using only a general representation. 
- If we look at another example:

<img src="https://drive.google.com/uc?export=view&id=1aFdmayzExrbRA2R-wffYo1eUPS-9Wfyf">

- We see that in one example, a circled dot is right at the place of the red X, so that’s super relevant as an answer to the query, but some other results from the retrieval are not exactly the nearest neighbors for the search.
- Another example:

<img src="https://drive.google.com/uc?export=view&id=1Nz-igEwWJu321CJIOuH3ehspMbkBLYPk">

- There’s also the trivial problem of a completely irrelevant query, such as “What has Michael Jordan done for us this year?”
- Projecting the UMAP figure for the results retrieved for this will also show that the retrieved chunks are from all over the place, as they have no connection to the query.

<img src="https://drive.google.com/uc?export=view&id=1kNJDL6AavdUMgF-KyTHmoobrqa6UTxFJ">

- A query like this will retrieve only distractor chunks from the Vector DB, which will have nothing to do with Michael Jordan, and will hence be difficult to understand and debug from an application perspective or from the developer’s perspective.
- So we need a way to deal with irrelevant queries as well as irrelevant results.
- In the next section, we will look at a technique to improve the quality of the queries using LLMs, using a technique called Query Expansion.

## ***4 - Query Expansion***

- The field of Information Retrieval has been around for a while as a sub-field of Natural Language Processing, and there’s many approaches to improving the relevancy of query results.
- But what’s new is we now have powerful LLMs, and we can use those to augment and enhance the queries that we send to our vector-based retrieval system to get better results.
- The first type of Query Expansion we’ll talk about is Expansion with Generated Answers.
- The way this works is, take your query, pass it to an LLM which you prompt to generate a hypothetical or imagined answer to your query, then concatenate your query with the imagined answer and use that as the new query to pass to your retrieval system.
- Then you return your query results as normal.

<img src="https://drive.google.com/uc?export=view&id=1REf-l5AcuJgt4OVRiSDjnOQSuZApBEee">

- Let’s take a look at how this works in practice.
- To enable Query Expansion with Generated Answers, here’s the code:

```
def augment_query_generated(query, model="gpt-3.5-turbo"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Provide an example answer to the given question, that might be found in a document like an annual report."
        },
        {"role": "user", "content": query}
    ] 

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "Was there significant turnover in the executive team?"
hypothetical_answer = augment_query_generated(original_query)

joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))
```

- The objective is to generate a hypothetical answer, and then we create our joint query, which is basically the original query prepending the hypothetical answer, both presented together to LLM as the joint query.
- This is what the joint query looks like:

<img src="https://drive.google.com/uc?export=view&id=1tWph-D6u2YiBwM8UeHDgARdkZ_SYnkb_">

- We see the original query, followed by the hypothetical answer.
- The Chroma collection is queried the usual way, we retrieve the relevant chunks and we use UMAP to project the approx position of the chunks in 2D space.

<img src="https://drive.google.com/uc?export=view&id=1gbFKjljOC6qVOKcc0-5vJ1xzYmnpVh6l">

- If we look at this picture, we see that the Red X is our original query, the Orange X is the new joint query containing the hypothetical answer, and as the close cluster of green circles shows (with the exception of one which is far away on the right), we get a nice cluster of relevant results which are close to the joint query but also somewhat close to the original query.
- So that was Query Expansion with Generated Answers.
- But there’s also another type of Query Expansion we can try - Query Expansion with Multiple Queries.

<img src="https://drive.google.com/uc?export=view&id=13qtc9WAIHvWou3GGmHyOv4y6dRDHEyB6">

- The way we use this method is we generate additional queries that might help in answering the question.
- We take the original query, pass it to the LLM, and ask it to generate several new queries related to the original query.
- Then we pass the new queries along with the original query to the Vector Database.
- That gives us results for each of these - the original and the new queries, and these results are passed into the RAG process.
- So let’s take a look at how this works in practice.

```
def augment_multiple_query(query, model="gpt-3.5-turbo"):
    messages = [`
        {
            "role": "system",
            "content": "You are a helpful expert financial research assistant. Your users are asking questions about an annual report."
            "Suggest up to five additional related questions to help them find the information they need, for the provided question."
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions."
        },
        {"role": "user", "content": query}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content
```

- The original query we have is “What were the most important factors that contributed to increases in revenue?”
- Some of the new multi queries we get are - “What were the most important factors that contributed to decreases in revenue?”, “What were the sources of revenue for the company?”, “How were sales and revenue distributed across different product lines or segments?” and so on.
- We feed all the queries to the Chroma collection, store the retrieved chunks, now we de-duplicate the retrieved chunks (since the same chunk could be retrieved by multiple queries as they’re related), and finally feed a selection from the overall chunks left to the LLM to generate the answer.

<img src="https://drive.google.com/uc?export=view&id=1tYo9naul7FYdmYc-xvsHvBEo8oUbvl2_">

- If we analyze the results from this UMAP projection, we see that the multi queries allow us to hit other parts of the database for answers, which we may not have been able to reach with our original query.
- The downside of this though, of course, is that now we have more retrieved results than we had originally (even after de-duplication of course). And we’re not sure if and which of these results are actually relevant to our query. 
- In the next section, using Cross-Encoder Re-ranking, we have a technique that allows us to actually score the relevancy of the returned results and use only the ones that we feel match our original query.

## ***5 - Cross-encoder Re-ranking***

- In the last section, we looked at how to improve the retrieval results by augmenting the query we send by passing it through an LLM.
- Now we’re going to use a technique called Cross-encoder Re-ranking to score the relevancy of our retrieved results, for the query we sent.
- Re-ranking is a way to order & score retrieved results according to their relevance to a query.
- Let’s look at how this works underneath.

<img src="https://drive.google.com/uc?export=view&id=1t_uLnXQoBF3coLOx7obwZXR7HOrmH4Tx">

- In Re-ranking, after we retrieve the results for a particular query, we pass the results along with our query to a re-ranking model. 
- This allows us to re-rank the output so that the most relevant results have the highest rank. 
- Another way to think about this is - your re-ranking model scores each of the results conditioned on the query, and those with the highest score are the most relevant.
- Then we can just select the top ranking results as the most relevant to our query.
- Let’s look at how this is done in practice.

```
query = "What has been the investment in research and development?"
results = chroma_collection.query(query_texts=query, n_results=10, include=['documents', 'embeddings'])

retrieved_documents = results['documents'][0]

for document in results['documents'][0]:
    print(word_wrap(document))
    print('')

from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)
print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o+1)
```

- One use of re-ranking is to get more information out of the long tail of query results.
- Let’s take another look at the query “What has been the investment in research and development?”
- Usually we’ve been asking for 5 results in return for our particular query, but now we’ll ask for 10.
- That means we’ll get a longer tail of possibly useful results.
- Again, we see that we get the same first 5 results, but we also get 5 new results, which might have relevant information to question.
- The trick is to figure out which of these results are actually relevant to our specific query instead of just being the nearest neighbors in embedding space. 
- The way we do that is through a Cross-encoder Re-ranking.
- Let’s first understand what exactly a Cross-encoder is.

<img src="https://drive.google.com/uc?export=view&id=1SA2HipMmc1DrjCi0_q2e_U51RIFImu1q">

- Sentence Transformers are made up of two kinds of models.
- Some Encoder models are in the category of “Bi-encoders”, where two separate queries can be encoded separately, and we can then use these two different outputs to compute a Cosine Similarity between them.
- In contrast, a “Cross-encoder” takes both the queries together, and passed them internally through a Classifier Neural Network, which outputs a score.
- In this way, a Cross-encoder can be used to score our retrieved results by passing our query and each retrieved document and scoring them using the Cross-encoder.

<img src="https://drive.google.com/uc?export=view&id=1gsuidho2Ftv-xxNO9bJBl5VDWP9TLxBo">

- So we use the Cross-encoder by passing in the original query and each one of the retrieved documents, and using the resulting score as a relevancy or ranking score for our retrieved results.

```
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)
print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o+1)
```

- We simply create a list of pair lists, where each pair is the query with one of the retrieved documents, and we pass on this list of pair lists to the Cross-encoder to return scores.
- When we do this, we see that the values of the similarity scores are not necessarily in the same order as the order in which the retrieved documents are returned, which makes re-ranking necessary - we can now re-order on the basis of the Cross-encoder scores.
- This has the effect of scanning and extracting more of the information from the long tail on relevancy to the original query.
- This can of course be combined with Query Expansion (either Generated Answers or Multi-queries) as well. In that case, we would first retrieve and de-duplicate the increased number of results from the Vector DB, and then that increased number of results could be subjected to Cross-encoder Re-ranking (with pairs of Original Query and each Retrieved Doc) to get the most relevant results first.

```
queries = [original_query] + generated_queries

results = chroma_collection.query(query_texts=queries, n_results=10, include=['documents', 'embeddings'])
retrieved_documents = results['documents']

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

unique_documents = list(unique_documents)

pairs = []
for doc in unique_documents:
    pairs.append([original_query, doc])

scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)
```

- One great thing about using a Cross-encoder model like this one, is that it’s extremely lightweight and can be run completely locally.
- So in this section, we learnt how to use a Cross-encoder for re-ranking, and we’ve applied it both to a single query and also to an augmented query (with query expansion) to filter those results which are relevant to the original query.
- This is a powerful technique worth experimenting with. It’s a good idea to try and understand, and get an intuition for how the re-ranking score might change depending on the query, because the Cross-encoder Re-ranker can emphasize different parts of your query than an Embedding model.
- So the ranking it provides is much more conditional on the specific query than what is naïvely returned by the retrieval system.
- In the next section, we’ll cover Query Adaptors, which are a way to directly augment the Query Embedding itself, using user feedback or other data to get better query results.

## ***6 - Embedding Adaptors***



## ***7 - Other Techniques***

- Embedding-space retrieval is still a very active area of research, and there’s a lot of other techniques we should be aware of.
- For example, we can fine-tune the embedding model directly using the same type of data that we used in the Embedding Adaptors section.
- There’s also been some good results published recently, in fine-tuning the LLM itself to expect retrieved results and reason about them.
- A couple of such papers published are:
- 1 - RA-DIT: Retrieval-Augmented Dual Instruction Tuning
- 2 - InstructRetro: Instruction-tuning post Retrieval-Augmented Pretraining
- We can also experiment with a more complicated Embedding Adaptor model using a full-blown neural network or even a Transformer layer.
- Similarly, we can use a more complex relevance scoring model rather than just using the Cross Encoder re-ranking described in that section.
- Finally, an often overlooked piece is that the quality of the retrieved results often depends on the way the data is chunked before it’s stored in the retrieval system itself.
- There’s a lot of experimentation going on about using deep models including Transformers for optimal and intelligent chunking.
- That wraps up the course - in this course, we covered the basics of Retrieval-Augmented Generation using embeddings-based retrieval. 
- We looked at how we can use LLMs to augment and enhance our queries to produce better retrieval results.
- We looked at how we can use a Cross-Encoder model for Re-ranking to score the retrieved results for relevancy.
- And we looked at how we can train an Embedding Adaptor using data from human feedback about relevancy to improve our query results.
- Finally we covered some of the more exciting work that’s ongoing right now in the research literature around improving retrieval for AI applications.

**The End!**

***WIP - More Notes Incoming!***
