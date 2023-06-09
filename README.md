
# Fuzzy Matching EIDL with PPP
## log
4/24/2023
While the task associated with this repository is completed so far, this is still an active repository with efforts to improve its syntaxes and efficiency.
## Installation
The repository is stable at Python [3.10.0](https://www.python.org/downloads/release/python-3100/)
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages:
```bash
pip install pandas
pip install numpy
pip install scikit-learn
pip install rapidfuzz
pip install sparse_dot_topn
```
## Usage
Inspired by [this medium article](https://towardsdatascience.com/fuzzy-matching-at-scale-84f2bfd0c536) and [this blog post](https://bergvca.github.io/2017/10/14/super-fast-string-matching.html), the following programs aim at performing fast fuzzy matching between the company names and addresses in the [EIDL Advance dataset until Dec 2020](https://data.sba.gov/dataset/covid-19-eidl-advance) and the clean PPP dataset.

A total of 3 files are relevant (Run sequentially in listed order):

```python
	1. EIDL_Advance_Preprocessing.py # merge and clean EIDL Advance Data
	2. EIDL_Advance_FuzzyMatching.py # perform fuzzy matching
	3. EIDL_Advance_merge_PPP.py # convert the labeled EIDL Advance dataset to long format mergeble with the existing PPP dataset
```
Note: The actual merging is actually performed in Stata with very few line of code and thus omitted here.

While the above articles have provided detailed explanations and walkthroughs about the algorithm, we will briefly go through the underlying mathematical concepts:

### TFIDF Vectorization
***
In Natural Language Processing, any corpus needs to have its features (words) extracted and processed first before any NLP models can be applied. A common text extraction technique is [TF-IDF vectorization](https://en.wikipedia.org/wiki/Tf%E2%80%93idf), which calculates the TF-IDF score of any word in a given passage. A more detailed discussion on TF-IDF can be found in [this article](https://towardsdatascience.com/text-vectorization-term-frequency-inverse-document-frequency-tfidf-5a3f9604da6d), but to give a quick summary:
- "TF" stands for "term frequency", which measures the frequency of a word (w) in a document (d). TF is defined as the ratio of a word’s occurrence ($w$) in a document ($d$) to the total number of words in a document.
$$ TF(w,d) = {occurences \:of \: word \: w \: in \: document \: d \over 
total \: number \: of \: words \: in \: document \: d }$$
- "IDF" stands for "inverse document frequency", which is the natural l reciprocal of the document frequency of document $d$ in the total corpus ($D$) (the collection of documents).
$$ IDF(w,D) = ln({total \: numbers \: of \: documents \: in \: D \over 
1+ number \: of \: documents \: that \: contains \: w})$$
We thus define TF-IDF as 
$$TFIDF(w, d, D) = TF(w, d) \cdot IDF(w, D)$$
What does this mean conceptually though? $TFIDF$  score assigns a numerical value to the importance of the word in an entire corpus, and the higher the score is, the more important the word is to the corpus. Its first component $TF(w, d)$ is higher the more times a word $w$ appears in a document $d$, but the $IDF(w, D)$ part diminishes the importance of word $w$ the more documents in $D$ contains $w$ as it gets smaller as more documents contain $w$, because this increases the likelihood that word $w$ is just a commonly used English word that does not have very useful particular meanings (words like $the$, $of$ etc). Note that the exact implementation of $TFIDF$ is slightly different from algorithm to algorithm (the log base in $IDF$, which in $sklearn$ is $e$, the natural log), but the basic idea should be the same.

So why does this matter this us? Now, the power of $TFIDF$ can be applied to small word texts as well. If we define $w$ as a character (or a subset of characters) in an address/name, $d$ as one entity of address/name, and $D$ as the entire columns of address/Name, then we can measure the importance of a subset of characters in an entire collection of names/addresses. Each entity (an address/a name) can then be represented by a vector with which each element representing the $TFIDF$ score of a subset of its characters.

### Cosine Similarity
***
Now that each entity is vectorized, we can compute the cosine similarities between each entity of the datasets we are trying to match together (PPP and EIDL Advance in our case). A detailed explanation on cosine similarity can be found [here](https://www.machinelearningplus.com/nlp/cosine-similarity/), we won't be spending additional time here elaborating the mathematical concepts. While in general vector-wise computation is fairly efficient in Python, the sheer number of dataset size we have means that we need to use something more efficient to perform these vector multiplication. We ended up using a [module](https://github.com/ing-bank/sparse_dot_topn) developed by ING Group which specializes in sparse matrix multiplications and top-n selections. In fact, this module is [specifically designed](https://medium.com/wbaa/even-faster-string-comparison-2778be7fe480) for efficient string comparisons with vectorized string entities by ING. 

Using the two techniques introduced above, we are able to perform fuzzy matching between the PPP dataset and EIDL dataset on names and addresses for a total duration of ~3 hrs (exact time not timed due to indiscretion :( but might be in the future), which is significantly faster than any traditional fuzzing matching technique commonly used both in Python and other data processing tools (a rough estimate on a dataset of our size is about at least 150 hours).


















## Contact:
If you have additional question, please contact Yuanzhan Gao at yg8ch@virginia.edu
