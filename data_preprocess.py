import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sms_data = pd.read_table('SMSSpamCollection', sep = '\t', header = None, 
                                                names = ['label', 'message'])

print(sms_data.shape)

sms_data['label'] = sms_data.label.map({'ham':0, 'spam':1})

print(sms_data.head())

documents = list(sms_data['message'])
# print(documents[0:5]) 

count_vector = CountVectorizer()
count_vector.fit(documents)
print(count_vector.get_feature_names())