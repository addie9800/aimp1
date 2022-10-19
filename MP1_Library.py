# =============================================================================
# Mini Project 1: Emotion and Sentiment Classification of Reddit Posts
# This is the Library file, which contains all methods that will serve on 
# solving the questions of Mini-Project 1 of Artificial Intelligent COMP 472
# =============================================================================

#%% Importation of the libraries ----------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pandas as pd
from nltk import word_tokenize
from tqdm import tqdm
import string
import copy
from gensim import downloader

#%% Class for the Mini Project one --------------------------------------------

class MP1:

    '''
    This class will contain several methods for solving the Mini-Project 1 in Artificial 
    Intelligence COMP 472.
    This has being programed to answer part one, two and three. In fact, you can create
    an instance of this class by referencing the data set, the type_vectorize and the model. Namely
    if you choose a WordEmbeddings (WE) as type_vectorizer you shloud enter a model otherwise not. 

    Attributes:
        data: A list that contains the data set (input and output)
        type_vectorize: A string  which define the method that will be use to turn the input in array of numebers. 
            - CV for CountVectorize: make sentences as a one-hot vector.
            - TFIDF for TfidfTransformer: The goal of using tf-idf instead of CountVectorize is to scale down 
            the impact of tokens that occur very frequently in a given corpus and that are hence empirically less 
            informative than features that occur in a small fraction of the training corpus
            - WE for WordEmbeddings: will turn each sentence in a vector up to 300 values
        model: A string that will define the template for the wordEmbeddings method.
    
    ''' 

    def __init__(self, data = list(), type_vectorize = "CV", model = "word2vec-google-news-300"):
        self.data = data
        self.type_vectorize = type_vectorize
        self.model = model
        if not isinstance(data, list):
            raise TypeError("Second argument of TypedList must "
                  "be a list.")
        if not isinstance(type_vectorize, str):
            raise TypeError("Second argument of TypedList must "
                      "be a string.")
        if not isinstance(model, str):
            raise TypeError("Second argument of TypedList must "
                  "be a string.")
        if type_vectorize not in ["CV", "TFIDF", "WE"]:
            raise TypeError("You must inter either CV, TFIDF or WE")
    
    def __str__(self): 
        X = MP1.__feature_extraction_CV(self)[0]
        out = ""
        if self.type_vectorize == "CV":
            out += "You will use the CountVectorize method for making the posts numerical\n"
            out += "The size of the vocabulary is: {}".format(X.shape[1])
        elif self.type_vectorize == "TFIDF":
            out += "You will use the TfidfTransformer method for making the posts numerical\n"
            out += "The size of the vocabulary is: {}".format(X.shape[1])
        else:
            out += "You will use the WordEmbeddings method for making the posts numerical\n"
            out += "you will use the model: " + self.model + " for embedding words"
        
        
        return out
    
    def extract_features(self):

        '''
        extract_features will take out emotions and 
        sentiments.
        return:
            - emotions: list of the different emotions present
            in the data set.
            - D_emotions: dictionary that attributes a number to each 
            emotions.
            - D_sentiments: dictionary that attributes a number to each 
            snetiments.
            - list_all_emotions: output of the emotions for each data set input.
            - list_all_sentiments: output of the sentiments for each data set input.
        ''' 

        #list all the different emotions and their occurences
        emotions = list(set([i[1] for i in self.data]))
        list_all_emotions = list([i[1] for i in self.data])
        counter_emotions = [list_all_emotions.count(i) for i in emotions]
        D_emotions = {emotion: number for emotion,number in zip(emotions,counter_emotions)}

        #list all the different sentiments and their occurences
        sentiments = list(set([i[2] for i in self.data]))
        list_all_sentiments = list([i[2] for i in self.data])
        counter_sentiments = [list_all_sentiments.count(i) for i in sentiments]
        D_sentiments = {sentiment:number for sentiment,number in zip(sentiments,counter_sentiments)}
        
        return emotions, D_emotions, D_sentiments, list_all_emotions, list_all_sentiments
        
    def show_data_bar(self):

        '''
        show_data_bar will plot a histogram of the repartition of the input regarding
        their emotions and sentiments.
        '''
        
        emotions, D_emotions, D_sentiments = MP1.extract_features(self)[0:3]
        fig1 , (ax1 , ax2) = plt.subplots(1 , 2)
        ind = np.arange(len(emotions))
        width = 0.75
        ax1.barh(ind, D_emotions.values(), width, color = 'c')
        for i, v in enumerate(D_emotions.values()):
            ax1.text(v + 3, i - 0.25, str(v), color='red', fontweight='bold')
        ax1.set_yticks(ind+width/6)
        ax1.set_yticklabels(D_emotions.keys(), minor=False)
        ax1.set_ylabel('Emotions')
        ax1.set_xlabel('Occurences number')
        ax1.set_title('Histogram counting emotions throught posts')
       
        ax2.bar(D_sentiments.keys(), D_sentiments.values(), color = 'c')
        for i,v in enumerate(D_sentiments.values()):
            ax2.text(i, v + 3, str(v), color='red', fontweight='bold')
        ax2.set_xlabel('Sentiments')
        ax2.set_ylabel('Occurences number')
        ax2.set_title('Histogram counting sentiments throught posts')
        
    def show_data_pie(self):

        '''
        show_data_pie will plot a pie diagram of the repartition of the input regarding
        their emotions and sentiments.
        '''

        emotions, D_emotions, D_sentiments = MP1.extract_features(self)[0:3]
        fig1 , (ax1 , ax2) = plt.subplots(1 , 2)
        ax1.pie(D_emotions.values(), labels = D_emotions.keys(), startangle=90, autopct='%1.1f%%')
        ax1.axis('equal')
        ax1.set_title('Histogram counting emotions throught posts')
                
        ax2.pie(D_sentiments.values(),labels= D_sentiments.keys(), startangle=90, autopct='%1.1f%%')
        ax2.axis('equal')
        ax2.set_title('Histogram counting sentiments throught posts')
    
    def __feature_extraction_CV(self):

        '''
        __feature_extraction_CV is a private method that will preprocess the data 
        before applying a Machine Learning method. This method will use the 
        CountVectorize method to turn sentences into numbers.
        return:
            - X_train_emo: training inputs for emotion classification.
            - X_test_emo: testing inputs for emotion classification.
            - y_train_emo: training outputs for emotion classification.
            - y_test_emo: testing outputs for emotion classification.
            - X_train_sen: training inputs for sentiment classification.
            - X_test_sen: testing inputs for sentiment classification.
            - y_train_sen: training outputs for sentiment classification.
            - y_test_sen: testing outputs for sentiment classification.
            - le_dict_emo: dictionnary that list the emotions and their attributed number.
            - le_dict_sen: dictionnary that list the seniments and their attributed number.
        '''

        list_all_emotions, list_all_sentiments = MP1.extract_features(self)[3:]
        posts = np.array([i[0] for i in self.data])

        le_sen = preprocessing.LabelEncoder()
        y_sen = le_sen.fit_transform(list_all_sentiments)
        le_dict_sen = dict(zip(le_sen.classes_, le_sen.transform(le_sen.classes_)))
        
        le_emo = preprocessing.LabelEncoder()
        y_emo = le_emo.fit_transform(list_all_emotions)
        le_dict_emo = dict(zip(le_emo.classes_, le_emo.transform(le_emo.classes_)))
        
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(posts)
        
        size_vocabulary = len(vectorizer.get_feature_names()) # there are 30449 words in our vocabulary
        
        #We split the data into 80% for training and 20% for testing
        X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(X,y_emo, test_size = 0.2, random_state=0)
        X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(X,y_sen, test_size = 0.2, random_state=0)

        return X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen, le_dict_emo, le_dict_sen

    def __feature_extraction_TFIDF(self):

        '''
        __feature_extraction_TFIDF is a private method that will preprocess the data 
        before applying a Machine Learning method. This method will use the 
        TfidfTransformer method to turn sentences into numbers.
        return:
            - X_train_emo: training inputs for emotion classification.
            - X_test_emo: testing inputs for emotion classification.
            - y_train_emo: training outputs for emotion classification.
            - y_test_emo: testing outputs for emotion classification.
            - X_train_sen: training inputs for sentiment classification.
            - X_test_sen: testing inputs for sentiment classification.
            - y_train_sen: training outputs for sentiment classification.
            - y_test_sen: testing outputs for sentiment classification.
            - le_dict_emo: dictionnary that list the emotions and their attributed number.
            - le_dict_sen: dictionnary that list the seniments and their attributed number.
        '''

        list_all_emotions, list_all_sentiments = MP1.extract_features(self)[3:]
        posts = np.array([i[0] for i in self.data])
        le_sen = preprocessing.LabelEncoder()
        y_sen = le_sen.fit_transform(list_all_sentiments)
        le_dict_sen = dict(zip(le_sen.classes_, le_sen.transform(le_sen.classes_)))
        
        le_emo = preprocessing.LabelEncoder()
        y_emo = le_emo.fit_transform(list_all_emotions)
        le_dict_emo = dict(zip(le_emo.classes_, le_emo.transform(le_emo.classes_)))
        
        vectorizer=CountVectorizer()
        transformer = TfidfTransformer()
        X_tfidf = transformer.fit_transform(vectorizer.fit_transform(posts)) 
        
        size_vocabulary = len(vectorizer.get_feature_names()) # there are 30449 words in our vocabulary
        
        #We split the data into 80% for training and 20% for testing
        X_train_emo, X_test_emo, y_train_emo, y_test_emo = train_test_split(X_tfidf,y_emo, test_size = 0.2, random_state=0)
        X_train_sen, X_test_sen, y_train_sen, y_test_sen = train_test_split(X_tfidf,y_sen, test_size = 0.2, random_state=0)
        
        return X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen, le_dict_emo, le_dict_sen

    def __feature_extraction_WE(self):
        
        '''
        __feature_extraction_WE is a private method that preprocesses the data if type_vectorize='WE' was passed on init. The chosen model is
        loaded and the averaged embedding vectors are calculated based on the model passed as input.
        returns:
            - training numpy array of embeddings
            - test numpy array of embeddings
            - training numpy array of emotions
            - test numpy array of emotions
            - training numpy array of sentiments
            - test numpy array of sentiments
        '''
        
        # Set up the model, as defined on initialization, downloads automatically if not present
        print("Downloading the model, might take few minutes... ")
        model = downloader.load(self.model)
        # custom set of training set size
        training_set_size = 0.8
        # manual split into training and test set, shufffling the data for better results
        length = round(training_set_size * len(self.data))
        content_array = np.array(self.data)
        np.random.shuffle(content_array)
        content_training = content_array[:length]
        content_test = content_array[length:]

        # Tokenize using nltk package, using list for better performance

        tokens_training = list()
        emotions_training = list()
        sentiments_training = list()
        number_tokens_training = 0
        for item in tqdm(content_training):
            # for each item in the training data set split the post into an array, where each element is either a word or punctuation
            temp = word_tokenize(item[0])
            tokens_training.append(np.array(temp))
            # parallel save the corresponding labels in another array
            emotions_training.append(item[1])
            sentiments_training.append(item[2])
            number_tokens_training += len(temp)
        print("The number of tokens in the training set is: " + str(number_tokens_training))
        # repeat the process from above for the test set
        tokens_test = list()
        emotions_test = list()
        sentiments_test = list()
        for item in tqdm(content_test):
            temp = word_tokenize(item[0])
            tokens_test.append(np.array(temp))
            emotions_test.append(item[1])
            sentiments_test.append(item[2])

        # Calculate the hit and miss rate in the training set EXCLUDING punctuation and calculate the embeddings of each post as the average of the embeddings of each word

        averaged_list = list()
        total_count = 0
        hit_count = 0
        for post in tqdm(tokens_training):
            averaged_list.append(model.get_mean_vector(post, pre_normalize=True, post_normalize=False, ignore_missing=True))
            for word in post:
                if word in string.punctuation:
                    continue
                total_count += 1
                if model.__contains__(word):
                    hit_count += 1
        print("The number of tokens excluding punctuation in the training set is: " + str(total_count))
        print("The hit-rate of the training set is " + str(hit_count/total_count) + " the miss-rate is " + str((total_count - hit_count)/total_count))

        # Calculate the embeddings for each post as the average of the embeddings in each word in the test set

        averaged_list_test = list()
        for post in tqdm(tokens_test):
            averaged_list_test.append(model.get_mean_vector(post, pre_normalize=True, post_normalize=False, ignore_missing=True))
        averaged_array_test = np.array(averaged_list_test)
        averaged_array = np.array(averaged_list)
        
        return averaged_array, averaged_array_test, np.array(emotions_training), np.array(emotions_test), np.array(sentiments_training), np.array(sentiments_test)
    
    def MNB(self):

        '''
        MNB will use the Multinomial Naive Bayes algorithm to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
        '''

        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            raise ValueError("This method works only with CountVectorize (CV) and TfidfTransformer (TFIDF)")
        
        clf_MNB_emo = MultinomialNB()
        clf_MNB_sen = MultinomialNB()

        model_emo = clf_MNB_emo.fit(X_train_emo,y_train_emo)
        model_sen = clf_MNB_sen.fit(X_train_sen,y_train_sen)

        y_pred_MNB_emo = model_emo.predict(X_test_emo)
        y_pred_MNB_sen = model_sen.predict(X_test_sen)
        
        f1_emo = f1_score(y_test_emo, y_pred_MNB_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_MNB_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_MNB_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_MNB_sen)
        report_emo = classification_report(y_test_emo, y_pred_MNB_emo)
        report_sen = classification_report(y_test_sen, y_pred_MNB_sen)

        print("The f1 metric of the MNB algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the MNB algortihm for sentiments is: {}".format(f1_sen))
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen

    def Top_MNB(self):

        '''
        Top_MNB will use the Multinomial Naive Bayes algorithm with top parameters to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
            - grid_emo.best_estimator_: best estimator for Multinomial Naive Bayes 
            applying to these data for classiiying emotions.
            - grid_sen.best_estimator_: best estimator for Multinomial Naive Bayes 
            applying to these data for classiiying sentiments.
        '''

        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            raise ValueError("This method works only with CountVectorize (CV) and TfidfTransformer (TFIDF)")
            
        clf_MNB_emo = MultinomialNB()
        clf_MNB_sen = MultinomialNB()

        parameter = {'alpha':[0, 0.5, 1, 10]}
        grid_emo = GridSearchCV(estimator = clf_MNB_emo, param_grid = parameter)
        grid_emo.fit(X_train_emo,y_train_emo)
        
        grid_sen = GridSearchCV(estimator = clf_MNB_sen, param_grid = parameter)
        grid_sen.fit(X_train_sen,y_train_sen)
        
        y_pred_MNB_emo = grid_emo.predict(X_test_emo)
        y_pred_MNB_sen = grid_sen.predict(X_test_sen)
        
        f1_emo = f1_score(y_test_emo, y_pred_MNB_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_MNB_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_MNB_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_MNB_sen)
        report_emo = classification_report(y_test_emo, y_pred_MNB_emo)
        report_sen = classification_report(y_test_sen, y_pred_MNB_sen)

        print("The f1 metric of the MNB algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the MNB algortihm for sentiments is: {}".format(f1_sen))
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen, grid_emo.best_estimator_, grid_sen.best_estimator_
    
    
    def DT(self):

        '''
        DT will use the Decision Tree algorithm to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
        '''

        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, y_train_sen, y_test_sen = MP1.__feature_extraction_WE(self)
            X_train_sen = copy.deepcopy(X_train_emo)
            X_test_sen = copy.deepcopy(X_test_emo)
            
        # Create Decision Tree classifer object
        clf_tree_emo = DecisionTreeClassifier()
        clf_tree_sen = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf_tree_emo = clf_tree_emo.fit(X_train_emo,y_train_emo)
        clf_tree_sen = clf_tree_sen.fit(X_train_sen,y_train_sen)

        #Predict the response for test dataset
        y_pred_tree_emo = clf_tree_emo.predict(X_test_emo)
        y_pred_tree_sen = clf_tree_sen.predict(X_test_sen)

        f1_emo = f1_score(y_test_emo, y_pred_tree_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_tree_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_tree_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_tree_sen)
        report_emo = classification_report(y_test_emo, y_pred_tree_emo)
        report_sen = classification_report(y_test_sen, y_pred_tree_sen)

        print("The f1 metric of the Tree Decision algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the Tree Decision algortihm for sentiments is: {}".format(f1_sen))
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen

    def Top_DT(self):

        '''
        Top_DT will use the Decision Tree algorithm with top parameters to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
            - grid_emo.best_estimator_: best estimator for Decision Tree algorithm
            applying to these data for classiiying emotions.
            - grid_sen.best_estimator_: best estimator for decision Tree algorithm
            applying to these data for classiiying sentiments.
        '''

        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, y_train_sen, y_test_sen = MP1.__feature_extraction_WE(self)
            X_train_sen = copy.deepcopy(X_train_emo)
            X_test_sen = copy.deepcopy(X_test_emo)
            
        # Create Decision Tree classifer object
        clf_tree_emo = DecisionTreeClassifier()
        clf_tree_sen = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        parameters = {'criterion':['gini', 'entropy'], 'max_depth':[i for i in range(2,10)], 'min_samples_split':[i for i in range(2,10)]}
        grid_emo = GridSearchCV(estimator = clf_tree_emo, param_grid = parameters)
        grid_emo.fit(X_train_emo,y_train_emo)
        
        grid_sen = GridSearchCV(estimator = clf_tree_sen, param_grid = parameters)
        grid_sen.fit(X_train_sen,y_train_sen)
       
        y_pred_tree_emo = grid_emo.predict(X_test_emo)
        y_pred_tree_sen = grid_sen.predict(X_test_sen)
        
        f1_emo = f1_score(y_test_emo, y_pred_tree_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_tree_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_tree_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_tree_sen)
        report_emo = classification_report(y_test_emo, y_pred_tree_emo)
        report_sen = classification_report(y_test_sen, y_pred_tree_sen)

        print("The f1 metric of the Tree Decision algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the Tree Decision algortihm for sentiments is: {}".format(f1_sen))
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen, grid_emo.best_params_, grid_sen.best_params_
    
    def MLP(self):

        '''
        MLP will use the Multi-layer perceptron Classifier algorithm to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
        '''
     
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, y_train_sen, y_test_sen = MP1.__feature_extraction_WE(self)
            X_train_sen = copy.deepcopy(X_train_emo)
            X_test_sen = copy.deepcopy(X_test_emo)
        
        # Create Neural Network classifer object
        clf_MLP_emo = MLPClassifier(max_iter = 2)
        clf_MLP_sen = MLPClassifier(max_iter = 2)

        clf_MLP_emo = clf_MLP_emo.fit(X_train_emo,y_train_emo)
        clf_MLP_sen = clf_MLP_sen.fit(X_train_sen,y_train_sen)


        y_pred_MLP_emo = clf_MLP_emo.predict(X_test_emo)
        y_pred_MLP_sen = clf_MLP_sen.predict(X_test_sen)

        f1_emo = f1_score(y_test_emo, y_pred_MLP_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_MLP_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_MLP_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_MLP_sen)
        report_emo = classification_report(y_test_emo, y_pred_MLP_emo)
        report_sen = classification_report(y_test_sen, y_pred_MLP_sen)

        print("The f1 metric of the Multi-Layered Perceptron algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the Multi-Layered Perceptron algortihm for sentiments is: {}".format(f1_sen))
        
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen

    def Top_MLP(self):

        '''
        MLP will use the Multi-layer perceptron Classifier algorithm with top parameters to classify the data set.
        return:
            - conf_matrix_emo: confusion matrix for emotions.
            - conf_matrix_sen: confusion matrix for sentiments. 
            - report_emo: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying emotions.
            -  report_sen: object taht contains different evaluation metrics 
            to analyse the efficiency of the algorithm for classifying sentiments. 
            - grid_emo.best_estimator_: best estimator for Multi-layer perceptron Classifier algorithm
            applying to these data for classiiying emotions.
            - grid_sen.best_estimator_: best estimator for Multi-layer perceptron Classifier algorithm
            applying to these data for classiiying sentiments.
        '''

        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_CV(self)[:8]
        elif self.type_vectorize == "TFIDF":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, y_train_sen, y_test_sen = MP1.__feature_extraction_WE(self)
            X_train_sen = copy.deepcopy(X_train_emo)
            X_test_sen = copy.deepcopy(X_test_emo)
            
        # Create Neural Network classifer object
        clf_MLP_emo = MLPClassifier(max_iter = 2)
        clf_MLP_sen = MLPClassifier(max_iter = 2)

        parameters = {'activation':['logistic', 'tanh', 'relu', 'identity'], 'solver':['adam', 'sgd'], 
                      'hidden_layer_sizes':[(30,50),(10,10,10)]}
        
        grid_emo = GridSearchCV(estimator = clf_MLP_emo, param_grid = parameters)
        grid_emo.fit(X_train_emo,y_train_emo)
        
        grid_sen = GridSearchCV(estimator = clf_MLP_sen, param_grid = parameters)
        grid_sen.fit(X_train_sen,y_train_sen)
       
        y_pred_MLP_emo = grid_emo.predict(X_test_emo)
        y_pred_MLP_sen = grid_sen.predict(X_test_sen)
        
        f1_emo = f1_score(y_test_emo, y_pred_MLP_emo, average = 'weighted')
        f1_sen = f1_score(y_test_sen, y_pred_MLP_sen, average = 'weighted')
        conf_matrix_emo = confusion_matrix(y_test_emo, y_pred_MLP_emo)
        conf_matrix_sen = confusion_matrix(y_test_sen, y_pred_MLP_sen)
        report_emo = classification_report(y_test_emo, y_pred_MLP_emo)
        report_sen = classification_report(y_test_sen, y_pred_MLP_sen)

        print("The f1 metric of the Tree Decision algortihm for emotions is: {}".format(f1_emo))
        print("the f1 metric of the Tree Decision algortihm for sentiments is: {}".format(f1_sen))
        
        
        return conf_matrix_emo, conf_matrix_sen, report_emo, report_sen, grid_emo.best_params_, grid_sen.best_params_
    
    def displaydict(self):

        '''
        displaudict will print the dictionnary that assign the corresponding number the emotion and sentiment.
        It will help the reader when analysing the report of each classifier algorithms.
        '''

        if self.type_vectorize == "CV":
            le_dict_emo, le_dict_sen = MP1.__feature_extraction_CV(self)[8:]
            print("Dictionnary of the correponding numbers and emotions: \n {} \n".format(le_dict_emo))
            print("Dictionnary of the correponding numbers and sentiments: \n {} \n".format(le_dict_sen))
        elif self.type_vectorize == "TFIDF":
            le_dict_emo, le_dict_sen = MP1.__feature_extraction_TFIDF(self)[8:]
            print("Dictionnary of the correponding numbers and emotions: \n {} \n".format(le_dict_emo))
            print("Dictionnary of the correponding numbers and sentiments: \n {} \n".format(le_dict_sen))
        else:
            print("No need of dictionaries because labels were not encoding")

        
    def analysis(self, matrix, report, output_name, reviewFile):

        '''
        analysis will put the confusion matrixes (emotion and sentiment) in a csv file. It will be
        easier to read the values.
        '''

        le_dict_emo, le_dict_sen = MP1.__feature_extraction_CV(self)[8:]
        if np.shape(matrix)[0] == 28:
            mat_emo = pd.DataFrame(matrix, columns = le_dict_emo.keys(), index = le_dict_emo.keys())
            mat_emo.to_csv(output_name + "_conf_emo.csv")
        else:
            mat_sen = pd.DataFrame(matrix, columns = le_dict_sen.keys(), index = le_dict_sen.keys())
            mat_sen.to_csv(output_name + "_conf_sen.csv")
        
        reviewFile.write(report)
