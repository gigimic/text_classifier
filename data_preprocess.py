import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
sms_data = pd.read_table('SMSSpamCollection', sep = '\t', header = None, 
                                                names = ['label', 'message'])

print(sms_data.shape)

sms_data['label'] = sms_data.label.map({'ham':0, 'spam':1})

print(sms_data.head())

documents = list(sms_data['message'])
# print(documents[0:5]) 

count_vector = CountVectorizer()

# this feature extraction library converts to lower case, removes punctuations
# and remove stop words

count_vector.fit(documents)
words = count_vector.get_feature_names()
print(len(words))

doc_array = count_vector.transform(documents).toarray()
frequency_matrix =pd.DataFrame(doc_array, columns = words)
print(frequency_matrix.head())
print(frequency_matrix.shape)

X = sms_data['message']
y = sms_data['label']

X_train, X_test, y_train, t_test = train_test_split(X, y,  random_state = 1)
print('Total number of rows: {}'.format(sms_data.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))
