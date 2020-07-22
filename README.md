![Tests](https://github.com/github/CodeSearchNet/workflows/Smoke%20Test/badge.svg)
 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)  [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Weights-And-Biases](https://img.shields.io/badge/Weights%20&%20Biases-black.svg?logo=google-analytics)](https://app.wandb.ai/github/CodeSearchNet/benchmark)

[paper]: https://arxiv.org/abs/1909.09436

# Semantic Code Search (CodeSearchNet Challenge)

As a project for the CS-GY-6613 Artificial Intelligence course at NYU Tandon School of Engineering, this extension of the [CodeSearchNet Challenge](https://app.wandb.ai/github/codesearchnet/benchmark) aims to improve upon the Neural Bag of Words baseline model. 


## Background

Searching for code on GitHub is currently limited to lexical search in which the user must anticipate the exact keywords or syntax in the code that they are looking for. But what if a user is unable to provide an exact literal match, and can provide a concept in the code? **Semantic search** seeks to improve upon lexical search by understanding the overall meaning of a query. Such systems take into account the user's intent by considering factors such as contextual meaning, synonyms, natural language queries, and more in order to provide relevant search results. For example, searching for code that performs the insertion sort algorithm without using the words "insertion" or "sort." A semantic search should return reasonable results even though there are no keywords in common between the query and the text. This has the potential to expedite the process of on-boarding new software engineers onto projects and bolster the discoverability of code in general. 


## Project Overview

[CodeSearchNet](https://arxiv.org/abs/1909.09436), is a collection of datasets and benchmarks that explore the problem of code retrieval using natural language. This research is a continuation of some ideas presented in this [blog post](https://githubengineering.com/towards-natural-language-semantic-code-search/) and is a joint collaboration between GitHub and the [Deep Program Understanding](https://www.microsoft.com/en-us/research/project/program/) group at [Microsoft Research - Cambridge](https://www.microsoft.com/en-us/research/lab/microsoft-research-cambridge/). The researchers aim to provide a platform for community research on semantic code search via the following: 

  1. Instructions for obtaining large corpora of relevant data
  2. Open source code for a range of baseline models, along with pre-trained weights
  3. Baseline evaluation metrics and utilities
  4. Mechanisms to track progress on a [shared community benchmark](https://app.wandb.ai/github/CodeSearchNet/benchmark) hosted by [Weights & Biases](https://www.wandb.com/)

The researchers hope that CodeSearchNet is a step towards engaging with the broader machine learning and NLP community regarding the relationship between source code and natural language. More context regarding the motivation for this problem is in this [technical report][https://arxiv.org/abs/1909.09436].

The initial CodeSearchNet Challenge implemented a range of baseline models including Neural Bag of Words (NBOW), Bidirectional RNN models, 1D Convolutional Neural Network, and Self-Attention. **For our project, we focused on improving NBOW by incorporating concatenation pooling.**

## Data Schema & Format

Data is stored in [jsonlines](http://jsonlines.org/) format.  Each line in the uncompressed file represents one example (usually a function with an associated comment). A prettified example of one row is illustrated below.

- **repo:** the owner/repo
- **path:** the full path to the original file
- **func_name:** the function or method name
- **original_string:** the raw string before tokenization or parsing
- **language:** the programming language
- **code:** the part of the `original_string` that is code
- **code_tokens:** tokenized version of `code`
- **docstring:** the top-level comment or docstring, if it exists in the original string
- **docstring_tokens:** tokenized version of `docstring`
- **sha:** this field is not being used [TODO: add note on where this comes from?]
- **partition:** a flag indicating what partition this datum belongs to of {train, valid, test, etc.} This is not used by the model.  Instead we rely on directory structure to denote the partition of the data.
- **url:** the url for the code snippet including the line numbers

## Model Architecture

  The baseline models used ingest a parallel corpus of (`comments`, `code`) and learn to retrieve a code snippet given a natural language query.  Specifically, `comments` are top-level function and method comments (e.g. docstrings in Python), and `code` is an entire function or method. Throughout this repo, we refer to the terms docstring and query interchangeably.

  The query has a single encoder, whereas each programming language has its own encoder. The available encoders are Neural Bag of Words, RNN, 1D-CNN, Self-Attention (BERT), and a 1D-CNN+Self-Attention Hybrid.

  The diagram below illustrates the general architecture of our baseline models:

  ![alt text](images/architecture.png "Architecture")


## Neural Bag of Words
Since this project focuses on a change to Neural Bag of Words, here we discuss it in more detail.

### a. Bag of Words Model

In general, Bag of Words is a method of extracting features from text which can be used in machine learning algorithms. It is called a "bag" of words because information about grammar and word order are discarded; it measures whether known words occur in a given text, and not where they occur. Each sentence or document is represented as a vector which contains a vocabulary of known words and a measure of the presence and the number of these known words in the text at hand. 

Consider the following example provided in [this](https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428) article. Given the following corpus:
* *"It was the best of times"*
* *"It was the worst of times"*
* *"It was the age of wisdom"*
* *"It was the age of foolishness"*

To vectorize our documents, we count how many times each word appears:

| Document | it | was | the | best | worst | age | of | times | wisdom | foolishness |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
*"It was the best of times"* | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 
*"It was the worst of times"* | 1 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 0 |
*"It was the age of wisdom"* | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 0 |
*"It was the age of foolishness"* | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 |

The occurence of words in a text can be scored using a variety of other methods as well: for example, with a boolean value (0 if absent, 1 if present), by calculating the frequency that each word appears in a document out of all the words in the document, or with TF-IDF (Term Frequency - Inverse Document Frequency).


### b. Neural Bag of Words 
The Neural Bag of Words baseline model used by CodeSearchNet is a fully connected feed forward network that serves as the architecture for creating token embeddings of the code and query token sequences created by the Bag of Words model. The token embeddings are then combined into a sequence embedding using a pooling function. The resulting embedding representations of the code and query now exist in a joint vector space; the code embedding that is "closest" to a given query embedding is likely to be the best semantic match as determined by NBOW.

<p align="center">
<img src="images/embeddings.png"  alt="drawing" width="450"/>
</p>

Figure from the CodeSearchNet Challenge [paper][https://arxiv.org/abs/1909.09436].

The authors of the original CodeNetChallenge implemented weighted mean pooling, but here we implement concatenation pooling which combines the mean and max pooling into one:

<img src="images/concat_code.png"  alt="drawing" width="800"/>

## Setup

A fellow student was kind enough to create a [guide](https://github.com/aobject/NYU-AI-Project-03) for setting up the environment for CodeSearchNet using a Ubuntu virtual machine (with a Nvidia P4000 GPU) and Docker. The following commands were used to setup the Docker container, and train and evaluate the model. Benchmarks were submitted to [Weights and Biases](https://app.wandb.ai/github/codesearchnet/benchmark) for detailed metrics and submission to the CodeSearchNet Challenge leaderboard.

  ```bash
  # clone this repository
  git clone https://github.com/github/CodeSearchNet.git
  cd CodeSearchNet/
  # download data (~3.5GB) from S3; build and run the Docker container
  script/setup
  # this will drop you into the shell inside a Docker container
  script/console
  # optional: log in to W&B to see your training metrics,
  # track your experiments, and submit your models to the benchmark
  wandb login

  # verify your setup by training a tiny model
  python train.py --testrun
  # see other command line options, try a full training run with default values,
  # and explore other model variants by extending this baseline script
  python train.py --help
  python train.py

  # generate predictions for model evaluation
  python predict.py -r github/CodeSearchNet/0123456 # this is the org/project_name/run_id
  ```

## Licenses

The licenses for source code used as data for this project are provided with the [data download](#downloading-data-from-s3) for each language in `_licenses.pkl` [files](resources/README.md#directory-structure).

This code and documentation for this project are released under the [MIT License](LICENSE).
