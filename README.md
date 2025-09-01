# Natural Language Processing (NLP) Projects Portfolio

A collection of hands-on fundamental projects in Natural Language Processing techniques. This repository serves as a practical guide and portfolio for key NLP tasks, implemented from scratch and using modern libraries.

## Table of Contents

- [Text Cleaning & Frequency Counter](#text-cleaning--frequency-counter)
- [Sentiment Analysis](#sentiment-analysis)
- [Rule & Retrieval-Based Chatbot](#rule--retrieval-based-chatbot)
- [Named Entity Recognition](#named-entity-recognition)
- [Dictionary-Based Machine Translation](#dictionary-based-machine-translation)
- [Character-Level Text Generation](#character-level-text-generation)
- [Transformer-Based Machine Translation](#transformer-based-machine-translation)

## Projects

### [Text Cleaning & Frequency Counter](#text-cleaning--frequency-counter)
**File:** `Text_cleaning_frequency_counter.ipynb`  
#### Description
This project demonstrates the foundational step of any Natural Language Processing (NLP) pipeline: **text preprocessing**. It provides a practical implementation for cleaning raw text data and analyzing word frequency, both with and without common stop words. The notebook processes a sample of Edgar Allan Poe's poetry, showcasing how to transform raw text into a structured format suitable for analysis.

#### Key Features & Functions
- **`text_cleaner(text)`**: Cleans and tokenizes input text by:
  - Removing all punctuation
  - Converting to lowercase
  - Tokenizing into individual words
- **`word_frequency_counter(text, remove_stopwords)`**: 
  - Counts word frequencies in cleaned text
  - Optional removal of English stopwords using NLTK
  - Returns a `Counter` object with word frequencies
- **Sample Analysis**: Includes a `main()` function that demonstrates the entire workflow on poetic text.

#### Concepts & Skills
- **Text Preprocessing**: The essential first step in NLP
- **Tokenization**: Breaking text into meaningful units using NLTK's `word_tokenize`
- **Stop Words**: Understanding and filtering common words (the, and, is)
- **Frequency Distribution**: Using `Collections.Counter` for textual analysis
- **NLTK Library**: Practical use of NLTK for tokenization and stopword management

#### Example Output
**With Stopwords:** [('the', 4), ('there', 2), ('no', 2), ('was', 2), ('and', 2)]

**Without Stopwords:** [('word', 2), ('deep', 1), ('darkness', 1), ('peering', 1), ('long', 1)]

This clearly shows how removing stopwords shifts the focus from grammatical structure to the more meaningful, content-bearing words in the text.

### [Sentiment Analysis](#sentiment-analysis)
**File:** `Sentiment_Analysis.ipynb`  
Classifies text (e.g., product reviews) into positive or negative sentiments using traditional Machine Learning models.  
**Concepts:** Sentiment classification, bag-of-words, TF-IDF, scikit-learn.

### [Rule & Retrieval-Based Chatbot](#rule--retrieval-based-chatbot)
**File:** `Rule_and_Retrieval_Based_chatbot.ipynb`  
Builds a simple chatbot that responds to user queries using pre-defined rules and by retrieving answers from a knowledge base.  
**Concepts:** Rule-based systems, similarity matching, dialog systems.

### [Named Entity Recognition](#named-entity-recognition)
**File:** `Named_Entity_Recognition.ipynb`  
Identifies and classifies named entities (e.g., persons, organizations, locations) in text.  
**Concepts:** Sequence labeling, spaCy, information extraction.

### [Dictionary-Based Machine Translation](#dictionary-based-machine-translation)
**File:** `dictionary_based_machine_translation.ipynb`  
A basic translation system that translates words by performing a direct dictionary lookup. Highlights the challenges of naive translation.  
**Concepts:** Lexical translation, challenges of MT (idioms, word sense).

### [Character-Level Text Generation](#character-level-text-generation)
**File:** `Character_level_Text_Generation.ipynb`  
Generates new text character-by-character using a Recurrent Neural Network (RNN) or LSTM model.  
**Concepts:** RNNs/LSTMs, text generation, deep learning.

### [Transformer-Based Machine Translation](#transformer-based-machine-translation)
**File:** `Transformer_Based_Machine_Translation.ipynb`  
Implements a state-of-the-art neural machine translation (NMT) system using the Transformer architecture.  
**Concepts:** Transformer model, attention mechanism, seq2seq learning.


## Tools & Libraries

- Python 3.8^
- TensorFlow / PyTorch (varies by notebook)
- OpenAI Gym
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook


