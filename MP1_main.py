# =============================================================================
# Mini Project 1: Emotion and Sentiment Classification of Reddit Posts.
# Main code 
# =============================================================================

#%% Importation of the libraries ----------------------------------------------
import json
from nltk import download
from MP1_Library import MP1 
import datetime
download('punkt')

#%% load data -----------------------------------------------------------------
#load the data
#open the file and extract the data
#convert json to array of arrays where each element is an array with [post, emotion, sentiment]
with open('goemotions.json') as file:
    data = json.load(file)

#Choose among these models for the Word embeddings method:
#model = "word2vec-google-news-300"
model = "glove-twitter-25"
# model = "glove-wiki-gigaword-200"

type_vectorize = "TFIDF" # You can choose between "WE", "TFIDF" and "CV"
mp1 = MP1(data, type_vectorize, model)
print(mp1)
mp1.displaydict()
#%% Data plot -----------------------------------------------------------------

mp1.show_data_pie()
mp1.show_data_bar()

#%% Multinomial Naive Bayes Classifier (default paramters)  -------------------
file_name_MNB = "performance_MNB" + '_' + type_vectorize

reviewFile_MNB = open(file_name_MNB + ".txt", "a")

conf_matrix_emo_MNB, conf_matrix_sen_MNB, report_emo_MNB, report_sen_MNB = mp1.MNB()
reviewFile_MNB.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_MNB.write("Multinomial Naive Bayes Classifier (default paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_MNB, report_emo_MNB, file_name_MNB, reviewFile_MNB)
reviewFile_MNB.write("\nMultinomial Naive Bayes Classifier (default paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_MNB, report_sen_MNB, file_name_MNB, reviewFile_MNB)

reviewFile_MNB.close()

#%% Multinomial Naive Bayes Classifier (top parameters)  ----------------------

file_name_Top_MNB = "performance_Top_MNB" + '_' + type_vectorize
reviewFile_Top_MNB = open(file_name_Top_MNB + ".txt", "a")

conf_matrix_emo_Top_MNB, conf_matrix_sen_Top_MNB, report_emo_Top_MNB, report_sen_Top_MNB, best_params_emo_Top_MNB, best_params_sen_Top_MNB = mp1.Top_MNB()
reviewFile_Top_MNB.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_Top_MNB.write("The best estimator for emotions across ALL searched params: {} \n".format(best_params_emo_Top_MNB))
reviewFile_Top_MNB.write("The best estimator for sentiments across ALL searched params: {} \n".format(best_params_sen_Top_MNB))
reviewFile_Top_MNB.write("Multinomial Naive Bayes Classifier (top paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_Top_MNB, report_emo_Top_MNB, file_name_Top_MNB, reviewFile_Top_MNB)
reviewFile_Top_MNB.write("\nMultinomial Naive Bayes Classifier (top paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_Top_MNB, report_sen_Top_MNB, file_name_Top_MNB, reviewFile_Top_MNB)

reviewFile_Top_MNB.close()

#%% Decision Tree Classifier (default paramters) ------------------------------

file_name_DT = "performance_DT" + '_' + type_vectorize
reviewFile_DT = open(file_name_DT + ".txt", "a")

conf_matrix_emo_DT, conf_matrix_sen_DT, report_emo_DT, report_sen_DT = mp1.DT()
reviewFile_DT.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_DT.write("Decision Tree Classifier (default paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_DT, report_emo_DT, file_name_DT , reviewFile_DT)
reviewFile_DT.write("Decision Tree Classifier (default paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_DT, report_sen_DT, file_name_DT , reviewFile_DT)

reviewFile_DT.close()

#%% Decision Tree Classifier (top paramters) ----------------------------------

file_name_Top_DT = "performance_Top_DT" + '_' + type_vectorize
reviewFile_Top_DT = open(file_name_Top_DT + ".txt", "a")

conf_matrix_emo_Top_DT, conf_matrix_sen_Top_DT, report_emo_Top_DT, report_sen_Top_DT, best_params_emo_Top_DT, best_params_sen_Top_DT = mp1.Top_DT()
reviewFile_Top_DT.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_Top_DT.write("The best estimator for emotions across ALL searched params: {} \n".format(best_params_emo_Top_DT))
reviewFile_Top_DT.write("The best estimator for sentiments across ALL searched params: {} \n".format(best_params_sen_Top_DT))
reviewFile_Top_DT.write("Decision Tree Classifier (top paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_Top_DT, report_emo_Top_DT, file_name_Top_DT , reviewFile_Top_DT)
reviewFile_Top_DT.write("Decision Tree Classifier (top paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_Top_DT, report_sen_Top_DT, file_name_Top_DT , reviewFile_Top_DT)

reviewFile_Top_DT.close()

#%% Multi-Layered Perceptron Classifier (default paramters) -------------------

file_name_MLP = "performance_MLP" + '_' + type_vectorize
reviewFile_MLP = open(file_name_MLP + ".txt", "a")

conf_matrix_emo_MLP, conf_matrix_sen_MLP, report_emo_MLP, report_sen_MLP = mp1.MLP()
reviewFile_MLP.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_MLP.write("Multi-Layered Perceptron Classifier (default paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_MLP, report_emo_MLP, file_name_MLP,  reviewFile_MLP)
reviewFile_MLP.write("Multi-Layered Perceptron Classifier (default paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_MLP, report_sen_MLP, file_name_MLP,  reviewFile_MLP)

reviewFile_MLP.close()
#%% Multi-Layered Perceptron Classifier (top paramters) -------------------

file_name_Top_MLP = "performance_Top_MLP" + '_' + type_vectorize
reviewFile_Top_MLP = open(file_name_Top_MLP + ".txt", "a")

conf_matrix_emo_Top_MLP, conf_matrix_sen_Top_MLP, report_emo_Top_MLP, report_sen_Top_MLP, best_params_emo_Top_MLP, best_params_sen_Top_MLP = mp1.Top_MLP()
reviewFile_Top_MLP.write("Date and Time: " + datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S') + "\n")
reviewFile_Top_MLP.write("The best estimator for emotions across ALL searched params: {} \n".format(best_params_emo_Top_MLP))
reviewFile_Top_MLP.write("The best estimator for sentiments across ALL searched params: {} \n".format(best_params_sen_Top_MLP))
reviewFile_Top_MLP.write("Multi-Layered Perceptron Classifier (top paramters) trained on emotions: \n\n")
mp1.analysis(conf_matrix_emo_Top_MLP, report_emo_Top_MLP, file_name_Top_MLP,  reviewFile_Top_MLP)
reviewFile_Top_MLP.write("Multi-Layered Perceptron Classifier (top paramters) trained on sentiments: \n\n")
mp1.analysis(conf_matrix_sen_Top_MLP, report_sen_Top_MLP, file_name_Top_MLP,  reviewFile_Top_MLP)
reviewFile_Top_MLP.close()

