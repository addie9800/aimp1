https://github.com/addie9800/aimp1/
# Mini-Project 1
## Emotion and Sentiment Classification of Reddit Posts
The main goal of this project was to use different machine learning algorithms to classify emotions and sentiments of reddit posts. We used embeddings and word vectorization to represent the data and used the Multinomial Naive Bayes, Decision Tree and Multilayer Perceptron algorithms. We produced reports which summarize the performance of each algorithm and data representation.

### Installation and Run of the Project Files

1. Download the project as a zip and extract it
2. Make sure the following Python modules are installed: gensim, numpy, matplotlib, sklearn, pandas, nltk, tqdm
3. Modify the MP1_main.py to perform the vectorization you want: set type_vectorize to
    - "CV" - Count Vecotorizer (Tasks 2.1 - 2.4)
    - "TFIDF" - Tf-Idf Vectorizer (Task 2.5)
    - "WE" - Word Embeddings (Task 3)
        - make sure to modify the model for task 3.8. Default is Google News 300
4. Run MP1_main.py with Python
5. The reports will be saved to the project directory

### Further Notes

For Task 3.8 we have decided to analyze how the embedding size influences the results of the machine learning algorithms. Apart from the Google News 300 model, as required, we have chosen the Twitter Glove 25 and Twitter Glove 200 models to do our analysis.

### Contributing

Adrian Breiding, Damien Martins Gomes and Alex Ye have contributed to this project.
