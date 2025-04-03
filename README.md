Spam-Ham Classifier
This project implements a Spam-Ham classifier using Natural Language Processing (NLP) techniques, specifically employing both Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) methods.​

Project Overview

​Implementing a Spam-Ham classifier using the Multinomial Naive Bayes algorithm with both Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) approaches is a well-established method in text classification. Multinomial Naive Bayes is particularly suited for discrete data, making it effective for document classification tasks where features represent word frequencies or occurrences. ​

Accuracy Score
A) By Using BOW and Stemming = 98.49
B) By Using TF-IDF and Lemmatization = 97.77

1. Data Preprocessing

Begin by loading and preprocessing your dataset. This involves cleaning the text data by removing punctuation, converting text to lowercase, and eliminating stopwords. Tokenization and lemmatization or stemming can further refine the text data.​

2. Feature Extraction

Apply both BoW and TF-IDF methods to convert text data into numerical features:​

Bag of Words (BoW): This approach creates a matrix of token counts, representing the frequency of each word in the corpus. While straightforward, BoW treats all words with equal importance and doesn't account for the context or significance of words within documents.​

TF-IDF: This method adjusts the frequency of words by their importance, reducing the weight of commonly used words and increasing the weight of words that are more unique to a document. TF-IDF often enhances the performance of text classification models by emphasizing informative terms. ​

3. Model Training and Evaluation

Split your dataset into training and testing sets. Train the Multinomial Naive Bayes classifier separately on the BoW and TF-IDF features. Evaluate each model's performance using metrics such as accuracy, precision, recall, and F1-score. It's common to observe that the TF-IDF approach yields higher accuracy compared to BoW, as TF-IDF provides a more nuanced representation of text data. ​

4. Repository Structure and Documentation

Organize your project repository to include the dataset, scripts for data preprocessing, feature extraction, model training, and evaluation. Provide detailed documentation explaining the methodologies used, the rationale behind choosing Multinomial Naive Bayes, and a comparative analysis of the results obtained from BoW and TF-IDF approaches. Including visualizations and tables can aid in illustrating performance differences and insights gained from the project.​
