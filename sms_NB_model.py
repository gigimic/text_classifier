import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sms_data = pd.read_table('SMSSpamCollection', sep = '\t', header = None, 
                                                names = ['label', 'message'])

print(sms_data.shape)

sms_data['label'] = sms_data.label.map({'ham':0, 'spam':1})

print(sms_data.head())

count_vector = CountVectorizer()

# this feature extraction library can convert to lower case, removes punctuations
# and remove stop words

X = sms_data['message']
y = sms_data['label']

# X_train, X_test, y_train, t_test = train_test_split(X, y, test_size = 0.3, random_state = 1)
X_train, X_test, y_train, y_test = train_test_split(sms_data['message'], sms_data['label'], test_size = 0.3, random_state = 1)

print('\n Total number of rows: {}'.format(sms_data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {} \n'.format(X_test.shape[0]))

# we are learning a vocabulary dictionary for the training data 
# and then transforming the data into a document-term matrix

training_data = count_vector.fit_transform(X_train)

# for the testing data we are only 
# transforming the data into a document-term matrix
test_data = count_vector.transform(X_test)

naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(test_data)


print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))