import gzip
import json
import matplotlib.pyplot as plt
import numpy as np
 
#open the file and extract the data

file = gzip.open('goemotions.json.gz')
jsonContent = file.read()
file.close()

#convert json to array of arrays where each element is an array with [post, emotion, sentiment]

content = json.loads(jsonContent)

#Count Sentiments in given dataset

neutral = 0
negative = 0
positive = 0
ambiguous = 0
for x in content:
    if x[2]== "neutral":
        neutral += 1
    elif x[2] == "negative":
        negative += 1
    elif x[2] == "positive":
        positive += 1
    elif x[2] == "ambiguous":
        ambiguous += 1
        
#Create and save Pie Chart for Sentiments

sentiChart = np.array([neutral, positive, ambiguous, negative])
sentiLabels = np.array(["Neutral", "Positive", "Ambiguous", "Negative"])
plt.pie(sentiChart, labels = sentiLabels)
plt.title("Distribution of Sentiments in the given dataset")
plt.savefig("Sentiment Distribution.pdf", format = 'pdf')

#Count emotions in given dataset as dictionary (key value pairs)

emotions = dict()
for x in content:
    currentEmotion = x[1]
    if currentEmotion in emotions:
        emotions[currentEmotion] += 1
    else:
        emotions[currentEmotion] = 1

#Create and save Bar Chart for emotions

emotionChart = np.array(list(emotions.values()))
emotionLabels = np.array(list(emotions.keys()))
fig = plt.figure()
fig.set_size_inches(15, 10)
x = np.arange(len(emotionLabels))
ax1 = plt.subplot()
ax1.set_xticks(x)
plt.bar(x,emotionChart)
ax1.set_xticklabels(emotionLabels, rotation = 90)
plt.subplots_adjust(bottom = 0.26)
plt.title("Distribution of Emotions in the given dataset")
plt.savefig("Emotion Distribution.pdf", format = 'pdf')
