# =============================================================================
# Mini Project 1: Emotion and Sentiment Classification of Reddit Posts
# =============================================================================

#%% Importation of the libraries ----------------------------------------------
import json
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
import datetime

#%% Class for the Mini Project one --------------------------------------------

class MP1:
    ''' This class will contain methods for solving the Mini Project 1 in
    Artificial Intelligence'''
    
    def __init__(self, data = list(), type_vectorize = "CV"):
        self.data = data
        self.type_vectorize = type_vectorize
        if not isinstance(data, list):
            raise TypeError("Second argument of TypedList must "
                  "be a list.")
        if not isinstance(type_vectorize, str):
            raise TypeError("Second argument of TypedList must "
                      "be a string.")
        if type_vectorize not in ["CV", "TFIDF"]:
            raise TypeError("You must inter either CV or TFIDF")
    
    def __str__(self):
        X = MP1.__feature_extraction(self)[0]
        out = ""
        if self.type_vectorize == "CV":
            out += "You will use the CountVectorize method for making the posts numerical\n"
        else:
            out+= "You will use the TfidfTransformer method for making the posts numerical\n"
        out += "The size of the vocabulary is: {}".format(X.shape[1])
        
        return out
    
    def extract_features(self):
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

        emotions, D_emotions, D_sentiments = MP1.extract_features(self)[0:3]
        fig1 , (ax1 , ax2) = plt.subplots(1 , 2)
        ax1.pie(D_emotions.values(), labels = D_emotions.keys(), startangle=90, autopct='%1.1f%%')
        ax1.axis('equal')
       
        ax1.set_title('Histogram counting emotions throught posts')
                
        ax2.pie(D_sentiments.values(),labels= D_sentiments.keys(), startangle=90, autopct='%1.1f%%')
        ax2.axis('equal')
        ax2.set_title('Histogram counting sentiments throught posts')
    
    def __feature_extraction(self):
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
        print(X_train_emo.shape)
        return X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen, le_dict_emo, le_dict_sen

    def __feature_extraction_TFIDF(self):
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
    
    def MNB(self):
        #Training part of the model with the Multinomial Naive Bayes classification. 
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        
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
        #Training part of the model with the Multinomial Naive Bayes classification. 
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        
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
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
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
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
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
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        # Create Neural Network classifer object
        clf_MLP_emo = MLPClassifier()
        clf_MLP_sen = MLPClassifier()

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
        if self.type_vectorize == "CV":
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction(self)[:8]
        else:
            X_train_emo, X_test_emo, y_train_emo, y_test_emo, X_train_sen, X_test_sen, y_train_sen, y_test_sen = MP1.__feature_extraction_TFIDF(self)[:8]
        # Create Neural Network classifer object
        clf_MLP_emo = MLPClassifier()
        clf_MLP_sen = MLPClassifier()

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
        if self.type_vectorize == "CV":
            le_dict_emo, le_dict_sen = MP1.__feature_extraction(self)[8:]
        else:
            le_dict_emo, le_dict_sen = MP1.__feature_extraction_TFIDF(self)[8:]
        
        print("Dictionnary of the correponding numbers and emotions: \n {} \n".format(le_dict_emo))
        print("Dictionnary of the correponding numbers and sentiments: \n {} \n".format(le_dict_sen))
        
    def analysis(self, matrix, report, output_name, reviewFile):
        le_dict_emo, le_dict_sen = MP1.__feature_extraction(self)[8:]
        if np.shape(matrix)[0] == 28:
            mat_emo = pd.DataFrame(matrix, columns = le_dict_emo.keys(), index = le_dict_emo.keys())
            mat_emo.to_csv(output_name + "_conf_emo.csv")
        else:
            mat_sen = pd.DataFrame(matrix, columns = le_dict_sen.keys(), index = le_dict_sen.keys())
            mat_sen.to_csv(output_name + "_conf_sen.csv")
        

        reviewFile.write(report)

