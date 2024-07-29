#!/usr/bin/env python
# coding: utf-8

# 
# ## Notebook content
# This notebook contains the steps and code to demonstrate the implementation of Semanting Chunking, to improve Retrieval Augumented Generation(RAG) in watsonx.ai in terms of both cost and performance. 
# 
# 
# ### About Retrieval Augmented Generation
# Retrieval Augmented Generation (RAG) is a versatile pattern that can unlock a number of use cases requiring factual recall of information, such as querying a knowledge base in natural language.
# 
# In its simplest form, RAG requires 3 steps:
# 
# - Index knowledge base passages (once)
# - Retrieve relevant passage(s) from knowledge base (for every user query)
# - Generate a response by feeding retrieved passage into a large language model (for every user query)
# 
# In this notebook, we will focus on improving #2, the retrieval of relvant passages. 
# Segmenting large-scale documents into coherent topic-based sections in digital content analysis is a significant challenge. Traditional methods(Recursive Character/Token/Sentence/Regex/Markdown Splitting) struggle with efficient splitting of content. Large documents, such as academic papers, lengthy reports, and detailed articles, are complex and contain multiple topics. These methods often miss subtle transitions or falsely identify them, leading to fragmented or overlapping sections.
# 
# Instead, we will use a method that attempts to analyze a document and group chunks based on topic. There are a few methods, each with it's own strengths and drawbacks. 
# ## Contents
# 
# This notebook contains the following parts:
# 
# - [Setup](#setup)
# - [Document data loading](#data)
# - [Build up knowledge base](#build_base)
# - [Foundation Models on watsonx](#models)
# - [Generate a retrieval-augmented response to a question](#predict)
# - [Summary and next steps](#summary)
# 

# <a id="setup"></a>
# ##  Set up the environment
# 
# Before you use the sample code in this notebook, you must perform the following setup tasks:
# 
# -  Create a <a href="https://cloud.ibm.com/catalog/services/watson-machine-learning" target="_blank" rel="noopener no referrer">Watson Machine Learning (WML) Service</a> instance (a free plan is offered and information about how to create the instance can be found <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/getting-started/wml-plans.html?context=wx&audience=wdp" target="_blank" rel="noopener no referrer">here</a>).
# 

# ### Install and import the dependecies

# In[ ]:


get_ipython().system('pip install "langchain==0.1.10" | tail -n 1')
get_ipython().system('pip install "ibm-watsonx-ai>=0.2.6" | tail -n 1')
get_ipython().system('pip install -U langchain_ibm | tail -n 1')
get_ipython().system('pip install wget | tail -n 1')
get_ipython().system('pip install sentence-transformers | tail -n 1')
get_ipython().system('pip install "chromadb==0.3.26" | tail -n 1')
get_ipython().system('pip install "pydantic==1.10.0" | tail -n 1')
get_ipython().system('pip install "sqlalchemy==2.0.1" | tail -n 1')


# In[ ]:


import os, getpass


# ### watsonx API connection
# This cell defines the credentials required to work with watsonx API for Foundation
# Model inferencing.
# 
# **Action:** Provide the IBM Cloud user API key. For details, see <a href="https://cloud.ibm.com/docs/account?topic=account-userapikey&interface=ui" target="_blank" rel="noopener no referrer">documentation</a>.

# In[ ]:


credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": getpass.getpass("Please enter your WML api key (hit enter): ")
}


# ### Defining the project id
# The API requires project id that provides the context for the call. We will obtain the id from the project in which this notebook runs. Otherwise, please provide the project id.
# 
# **Hint**: You can find the `project_id` as follows. Open the prompt lab in watsonx.ai. At the very top of the UI, there will be `Projects / <project name> /`. Click on the `<project name>` link. Then get the `project_id` from Project's Manage tab (Project -> Manage -> General -> Details).
# 

# In[ ]:


try:
    project_id = os.environ["PROJECT_ID"]
except KeyError:
    project_id = input("Please enter your project_id (hit enter): ")


# <a id="data"></a>
# ## Document data loading
# 
# Download the file with State of the Union.

# In[ ]:


import wget

filename = 'state_of_the_union.txt'
url = 'https://raw.github.com/IBM/watson-machine-learning-samples/master/cloud/data/foundation_models/state_of_the_union.txt'

if not os.path.isfile(filename):
    wget.download(url, out=filename)


# <a id="build_base"></a>
# ## Build up knowledge base
# 
# The most common approach in RAG is to create dense vector representations of the knowledge base in order to calculate the semantic similarity to a given user query.
# 
# In this basic example, we take the State of the Union speech content (filename), split it into chunks, embed it using an open-source embedding model, load it into <a href="https://www.trychroma.com/" target="_blank" rel="noopener no referrer">Chroma</a>, and then query it.

# In[ ]:


from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

loader = TextLoader(filename)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# The dataset we are using is already split into self-contained passages that can be ingested by Chroma.

# ### Create an embedding function
# 
# Note that you can feed a custom embedding function to be used by chromadb. The performance of Chroma db may differ depending on the embedding model used. In following example we use watsonx.ai Embedding service. We can check available embedding models using `get_embedding_model_specs`

# In[ ]:


from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs

get_embedding_model_specs(credentials.get('url'))


# In[ ]:


from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

embeddings = WatsonxEmbeddings(
    model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,
    url=credentials["url"],
    apikey=credentials["apikey"],
    project_id=project_id
    )
docsearch = Chroma.from_documents(texts, embeddings)


# #### Compatibility watsonx.ai Embeddings with LangChain
# 
#  LangChain retrievals use `embed_documents` and `embed_query` under the hood to generate embedding vectors for uploaded documents and user query respectively.

# In[ ]:


help(WatsonxEmbeddings)


# <a id="models"></a>
# ## Foundation Models on `watsonx.ai`

# IBM watsonx foundation models are among the <a href="https://python.langchain.com/docs/integrations/llms/watsonxllm" target="_blank" rel="noopener no referrer">list of LLM models supported by Langchain</a>. This example shows how to communicate with <a href="https://newsroom.ibm.com/2023-09-28-IBM-Announces-Availability-of-watsonx-Granite-Model-Series,-Client-Protections-for-IBM-watsonx-Models" target="_blank" rel="noopener no referrer">Granite Model Series</a> using <a href="https://python.langchain.com/docs/get_started/introduction" target="_blank" rel="noopener no referrer">Langchain</a>.

# ### Defining model
# You need to specify `model_id` that will be used for inferencing:

# In[ ]:


from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

model_id = ModelTypes.GRANITE_13B_CHAT_V2


# ### Defining the model parameters
# We need to provide a set of model parameters that will influence the result:

# In[ ]:


from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}


# ### LangChain CustomLLM wrapper for watsonx model
# Initialize the `WatsonxLLM` class from Langchain with defined parameters and `ibm/granite-13b-chat-v2`. 

# In[ ]:


from langchain_ibm import WatsonxLLM

watsonx_granite = WatsonxLLM(
    model_id=model_id.value,
    url=credentials.get("url"),
    apikey=credentials.get("apikey"),
    project_id=project_id,
    params=parameters
)


# <a id="predict"></a>
# ## Generate a retrieval-augmented response to a question

# Build the `RetrievalQA` (question answering chain) to automate the RAG task.

# In[ ]:


from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=watsonx_granite, chain_type="stuff", retriever=docsearch.as_retriever())


# ### Select questions
# 
# Get questions from the previously loaded test dataset.

# In[ ]:


query = "What did the president say about Ketanji Brown Jackson"
qa.invoke(query)


# ---

# <a id="summary"></a>
# ## Summary and next steps
# 
#  You successfully completed this notebook!.
#  
#  You learned how to answer question using RAG using watsonx and LangChain.
#  
# Check out our _<a href="https://ibm.github.io/watsonx-ai-python-sdk/samples.html" target="_blank" rel="noopener no referrer">Online Documentation</a>_ for more samples, tutorials, documentation, how-tos, and blog posts. 

# Copyright © 2023, 2024 IBM. This notebook and its source code are released under the terms of the MIT License.
