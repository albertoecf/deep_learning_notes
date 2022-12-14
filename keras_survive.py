#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

#def funcs
def get_title_from_name(name):
  if '.' in name:
    return name.split(',')[1].split('.')[0].strip()
  else:
    return "No title in name"

def title_classificator(x):
      title = x["Title"]
      if title in ['Capt', 'Col', 'Major']:
        return 'Officer'
      elif title in ['Jonkheer',"Don", "the Countess", "Dona", "Lady", "Sir"]:
        return "Royalty"
      elif title == "Mme":
        return "Mrs"
      elif title in ["Mlle", "Ms"]:
        return 'Miss'
      else:
        return title

#load
train_df = pd.read_csv('train.csv')

# Cleaning - Correcting
train_df.loc[train_df['Fare']>300, "Fare"] = train_df["Fare"].median()
# Cleaning - Feature eng
title = set([x for x in train_df.Name.map(lambda x: get_title_from_name(x))])
train_df['Title'] = train_df['Name'].map(lambda x: get_title_from_name(x))
train_df['Title'] = train_df.apply(title_classificator, axis=1)
# Cleaning - Completing
train_df['Age'].fillna(train_df['Age'].median(),inplace=True)
train_df['Fare'].fillna(train_df['Fare'].median(),inplace=True)
train_df['Embarked'].fillna("S", inplace=True)
train_df.drop(["Cabin",'Ticket'], axis=1, inplace=True )
del train_df['Name']
del train_df['PassengerId']
# Cleaning - Convert to number
train_df['Sex'].replace(('male','female'),(0,1),inplace=True)
train_df['Embarked'].replace( ('S','C','Q'),(0,1,2), inplace=True  )
train_df['Title'].replace( ('Mr','Miss','Mrs','Master','Dr','Rev','Officer','Royalty'),(0,1,2,3,4,5,6,7), inplace=True  )

# ML preprocessing
y = train_df['Survived']
x = train_df[['Pclass','Sex','Age','Fare','Embarked','Title']]
# Split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
# Initiate model
randomforest = RandomForestClassifier()
# Fit
randomforest.fit(x_train, y_train)
# Predict
y_pred = randomforest.predict(x_test)

# save model
file_name = 'testing.sav'
pickle.dump(randomforest, open(file_name,'wb'))

# %%
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout #Dense -> Hidden layer of the net

model = Sequential()
# model.add adds one "Dense" , one Hidden layer
model.add(Dense(32, activation='relu', input_shape=(6,)))
model.add(Dense(32, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#compile the model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#train the model
model_train = model.fit(x,y, epochs=100, batch_size=50, verbose=0, validation_split=0.1) 
#How accuracy changes with n epochs?
plt.plot(model_train.history['accuracy'], label='train')
plt.plot(model_train.history['val_accuracy'], label='test')
plt.title('Model accuracy')
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# %%
