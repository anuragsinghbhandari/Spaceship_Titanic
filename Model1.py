#importing 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

#reading training file
path ="train.csv"
data=pd.read_csv(path)

#data.columns
#data.head()

#turning nan into 0
data=data.fillna(0)

#target
y = data.Transported

#features
data_features = ["CryoSleep", "VIP", "Age", "VRDeck"]
X= data[data_features]


#building model
model = DecisionTreeClassifier(criterion="entropy",random_state=1)
model.fit(X,y)

#using on test file
test_path = "test.csv"
test_data=pd.read_csv(test_path)
test_data= test_data.fillna(0)
test_X = test_data[data_features]

#storing results into submission file 
ids = test_data['PassengerId'].tolist()
predictions = model.predict(test_X)
subdata = {'PassengerID': ids, 'Transported': predictions}
df = pd.DataFrame(subdata)
df.to_csv('submission.csv', index=False)

