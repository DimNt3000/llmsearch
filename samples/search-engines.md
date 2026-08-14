# How search engines work

A search engine has three moving parts: a crawler, an indexer, and a ranker.

The crawler (or spider) starts from a set of seed URLs, downloads each page,
extracts the links it contains, and adds new URLs to a queue. Polite crawlers
respect the robots.txt exclusion protocol, identify themselves with a
User-Agent string, and throttle their request rate so they do not overload the
sites they visit. Crawling the public web at scale is mostly an engineering
problem: deduplication, freshness scheduling, and trap avoidance.

The indexer turns each downloaded page into something searchable. Text is
extracted from the HTML, normalized (lowercased, tokenized, sometimes stemmed),
and stored in an inverted index: a map from each term to the list of documents
that contain it, along with the term frequency in each document. The inverted
index is what makes search fast — instead of scanning every document for a
query term, the engine looks the term up directly and gets back a postings
list.

The ranker decides the order of results. Classic lexical ranking functions
such as TF-IDF and BM25 score documents by how often the query terms appear in
them, discounted by how common the terms are across the whole collection.
Modern web search layers many more signals on top: link analysis (PageRank),
freshness, click feedback, and increasingly neural relevance models that
compare the meaning of the query and the document rather than their exact
words.

Small personal search engines follow the same architecture at a tiny scale: a
polite crawler feeding an inverted index, BM25 for ranking, and optionally a
language model on top for query understanding, summarization, and question
answering over the indexed content.
