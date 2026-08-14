# Retrieval-augmented generation (RAG)

Retrieval-augmented generation combines a search system with a large language
model. Instead of asking the model to answer from its training data alone, the
system first retrieves passages relevant to the question, then instructs the
model to answer using only those passages, citing them. This grounds the
answer in inspectable sources and sharply reduces hallucination.

A typical RAG pipeline has four stages. Ingestion splits documents into chunks
of a few hundred to a couple thousand characters — chunking matters because
retrieval works best when each unit of text is about one thing, and because
only a limited number of chunks fit into the model's context window. Indexing
stores the chunks in a retrieval system: a lexical index (BM25), a vector
database of embeddings, or both. Retrieval takes the user's question, finds
the top-k most relevant chunks, and optionally reranks them with a stronger
model. Generation feeds the question plus the retrieved chunks to the language
model with instructions to answer only from the provided context and to cite
which chunk supports each claim.

Two refinements are common. Query expansion asks the language model to rewrite
the user's question into several alternative phrasings before retrieval, which
helps when the user's vocabulary differs from the documents'. Rank fusion
(such as reciprocal rank fusion) merges the result lists from multiple queries
or multiple retrievers into one ranking.

The failure modes are instructive. If retrieval misses the relevant passage,
the model either refuses or hallucinates — so retrieval quality bounds answer
quality. If chunks are too large, the relevant sentence is diluted by noise;
too small, and the model lacks context. Citations make these failures visible:
an answer that cites its sources can be checked, an answer without sources has
to be trusted blindly.
