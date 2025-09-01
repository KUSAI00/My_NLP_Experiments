# Natural Language Processing (NLP) Projects Portfolio

A collection of hands-on fundamental projects in Natural Language Processing techniques. This repository serves as a practical guide and portfolio for key NLP tasks, implemented from scratch and using modern libraries.

## Table of Contents

- [Text Cleaning & Frequency Counter](#text-cleaning--frequency-counter)
- [Sentiment Analysis](#sentiment-analysis)
- [Rule & Retrieval-Based Chatbot](#rule--retrieval-based-chatbot)
- [Named Entity Recognition](#named-entity-recognition)
- [Dictionary-Based Machine Translation](#dictionary-based-machine-translation)
- [Character-Level Text Generation](#character-level-text-generation)

## Projects

### Text Cleaning & Frequency Counter
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

-----------

### Sentiment Analysis
**File:** `Sentiment_Analysis.ipynb`  

#### Description
A comprehensive sentiment analysis project that implements and compares multiple approaches to determine the emotional tone (positive, negative, neutral) of text. This notebook serves as a practical guide to both rule-based and machine learning-based sentiment analysis techniques, using movie reviews and product feedback as primary examples.

#### Key Features & Methods
- **Rule-Based Analysis**: 
  - **Lexicon-Based**: Uses NLTK's Opinion Lexicon for simple word-counting approach
  - **VADER**: Specialized rule-based model for social media/text sentiment
  - **TextBlob**: Simplified text processing and sentiment analysis
- **Machine Learning Approach**:
  - **LinearSVC Classifier**: Support Vector Machine with TF-IDF features
  - **Naive Bayes**: Traditional probabilistic classifier
  - **Custom Pipeline**: Full ML workflow from preprocessing to evaluation
- **Multi-Model Comparison**: Evaluates and compares performance across all approaches

#### Concepts & Skills
- **Sentiment Classification**: Three-class (positive/negative/neutral) text classification
- **Feature Engineering**: TF-IDF vectorization with n-grams
- **Model Evaluation**: Accuracy scores and detailed classification reports
- **Text Preprocessing**: Cleaning, tokenization, stopword removal
- **Cross-Validation**: Train-test splits with stratification
- **Multiple NLP Libraries**: NLTK, TextBlob, scikit-learn, VADER

#### Model Performance
**Evaluation Results:**
- VADER: Accuracy: 0.67
- TextBlob: Accuracy: 0.78
- LinearSVC: Accuracy: 0.44

#### Applications
- Product review analysis
- Social media monitoring
- Customer feedback processing
- Brand sentiment tracking
- Comparative study of sentiment analysis techniques

-----------

### Rule & Retrieval-Based Chatbot
**File:** `Rule_and_Retrieval_Based_chatbot.ipynb`  

#### Description
This project implements two distinct approaches to building chatbots: a **rule-based system** using pattern matching with regular expressions, and a **retrieval-based system** using TF-IDF vectorization and cosine similarity to find the most appropriate response from a predefined set of question-answer pairs.

#### Key Features & Approaches

**1. Rule-Based Chatbot (`simple_chatbot`)**
- Uses regular expressions to match user input patterns
- Handles specific conversation scenarios:
  - Greetings (hi, hello, hey)
  - Personal questions (name, identity)
  - Courtesy exchanges (thanks, goodbye)
  - Common queries (time, date, help)
- Returns predefined responses with random variation
- Simple and deterministic approach

**2. Retrieval-Based Chatbot (`RetrievalChatbot` class)**
- Uses TF-IDF vectorization to convert text to numerical features
- Employs cosine similarity to find the most relevant question-answer pair
- Includes similarity threshold (0.5) to handle unknown queries
- Preprocesses text by lowercasing and removing punctuation
- Learns from a predefined knowledge base of Q&A pairs

#### Concepts & Skills
- **Regular Expressions**: Pattern matching for text classification
- **TF-IDF Vectorization**: Converting text to numerical representations
- **Cosine Similarity**: Measuring text similarity in vector space
- **Information Retrieval**: Finding best matches from a knowledge base
- **Text Preprocessing**: Cleaning and normalizing input text
- **Chatbot Architecture**: Design patterns for conversational AI

#### Sample Knowledge Base
The chatbot is trained on 14 question-answer pairs covering:

- "hi": "Hello! How can I help you today?",
- "what's your name": "I'm a retrieval-based chatbot. You can call me RBot.",
- "how are you": "I'm just a program, but thanks for asking!",
- "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!"
  
#### Applications
- Customer service automation
- FAQ answering systems
- Educational assistants
- Entertainment and companionship
- Prototyping conversational interfaces

-----------

### Named Entity Recognition
**File:** `Named_Entity_Recognition.ipynb`  
#### Description
This project demonstrates Named Entity Recognition (NER) using two popular NLP libraries: **spaCy** and **NLTK**. NER is the task of identifying and classifying named entities in text into predefined categories such as persons, organizations, locations, dates, and monetary values. The notebook includes practical applications for financial news analysis and entity extraction.

#### Key Features & Approaches

**1. spaCy Implementation**
- Uses `en_core_web_sm` model for high-quality entity recognition
- Extracts entities with detailed labeling (ORG, DATE, GPE, MONEY, etc.)
- Visualizes entities with color-coded highlighting using `displacy`
- Includes specialized function for company/organization extraction

**2. NLTK Implementation**
- Uses `ne_chunk` with pre-trained models for entity recognition
- Combines part-of-speech tagging with named entity recognition
- Provides alternative approach to entity extraction
- Demonstrates compatibility with traditional NLP workflows

#### Entity Types Recognized
The models identify various entity types including:
- **ORG**: Organizations, companies, institutions
- **PERSON**: People's names
- **GPE**: Geopolitical entities (countries, cities, states)
- **DATE**: Absolute or relative dates or periods
- **MONEY**: Monetary values and currencies
- **PRODUCT**: Objects, vehicles, foods, etc.

#### Concepts & Skills
- **Named Entity Recognition**: Identifying and classifying entities in text
- **Statistical NLP**: Pre-trained models for entity detection
- **Text Visualization**: Color-coded entity highlighting with displacy
- **Library Comparison**: spaCy vs NLTK for NER tasks
- **Information Extraction**: Specific entity type filtering (e.g., companies only)

#### Applications
- Financial Analysis: Extract company names from news articles
- Content Categorization: Identify people, places, and organizations in documents
- Information Retrieval: Enhance search with entity recognition
- Data Mining: Extract structured information from unstructured text
- News Monitoring: Track mentions of specific entities

-----------
### Dictionary-Based Machine Translation
**File:** `dictionary_based_machine_translation.ipynb`  

#### Description
This project implements a simple **dictionary-based machine translation system** that translates text from English to French using a predefined word-to-word dictionary. It serves as a foundational example of machine translation, demonstrating the basic principles of lexical substitution while highlighting the limitations of naive translation approaches.

#### Key Features & Implementation

**Core Components:**
- **Bilingual Dictionary**: Predefined English-French word mappings (50+ entries)
- **Word Translation**: Direct word-to-word lookup with case handling
- **Sentence Translation**: Word-by-word translation of input sentences
- **Interactive Mode**: Real-time translation interface
- **Dynamic Dictionary**: Ability to add new translations during runtime

#### Concepts & Skills
- **Lexical Translation**: Word-level substitution approach
- **Dictionary-Based Systems**: Foundation of early machine translation
- **Text Processing**: Tokenization and word-by-word analysis
- **Interactive Systems**: User-friendly translation interface
- **Limitations of Naive MT**: Understanding translation challenges

-----------
### Character-Level Text Generation
**File:** `Character_level_Text_Generation.ipynb`  
#### Description
This project implements a **character-level text generation model** using a Recurrent Neural Network (RNN) with GRU layers. Trained on Shakespeare's complete works, the model learns to generate new text character by character, capturing the style, vocabulary, and structure of Elizabethan English. This demonstrates the power of neural networks in learning and reproducing complex linguistic patterns.

#### Model Architecture & Features

**Core Components:**
- **GRU-based RNN**: Gated Recurrent Unit layers for sequence modeling
- **Character-Level Processing**: Works at the character level (not words)
- **Embedding Layer**: 256-dimensional character embeddings
- **Vocabulary**: 66 unique characters including punctuation and symbols
- **Temperature Sampling**: Controlled randomness for text generation

**Technical Specifications:**
- **Vocabulary Size**: 66 characters
- **Embedding Dimension**: 256
- **GRU Units**: 256
- **Sequence Length**: 100 characters
- **Batch Size**: 64
- **Training Epochs**: 5

#### Concepts & Skills
- **Recurrent Neural Networks**: Understanding sequential data processing
- **Character-Level Modeling**: Working with text at the character level
- **Text Generation**: Autoregressive prediction of sequences
- **Temperature Sampling**: Controlling creativity vs. accuracy
- **TensorFlow/Keras**: Deep learning framework implementation
- **Text Preprocessing**: Handling and preparing text data for NLP

#### Training Details
- **Dataset**: Complete works of Shakespeare (1,115,394 characters)
- **Training Time**: ~45 seconds per epoch on standard hardware
- **Final Loss**: 1.56 after 5 epochs
- **Text Quality**: Coherent Shakespeare-style text with proper formatting

-----------

## Tools & Libraries

- Python 3.8^
- TensorFlow / PyTorch (varies by notebook)
- OpenAI Gym
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook


