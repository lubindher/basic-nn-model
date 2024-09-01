# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
A neural network can be used to solve a problem by Data collection and Preprocessing,choosing a appropriate neural network architecture ,Train the neural network using the collected and preprocessed data,Assess the performance of the trained model using evaluation metrics,Depending on the performance of the model, you might need to fine-tune hyperparameters or adjust the architecture to achieve better results,Once you're satisfied with the model's performance, you can deploy it to production where it can be used to make predictions on new, unseen data

For the problem statement we have dealt with , we have developed a neural network with three hidden layers. First hidden layer consists of 4 neurons ,second hidden layer with 8 neurons , third layer with 5 neurons . The input and output layer contain 1 neuron . The Activation Function used is 'relu'.

## Neural Network Model




## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name:LUBINDHER S
### Register Number:212222240056
```python
import pandas as pd

from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('MARKSDATA').sheet1

rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'int'})
df = df.astype({'OUTPUT':'int'})
df.head()

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)

Scaler = MinMaxScaler()

Scaler.fit(X_train)

X_train1 = Scaler.transform(X_train)

marks_data = Sequential([Dense(6,activation='relu'),Dense(1)])

marks_data.compile(optimizer = 'rmsprop' , loss = 'mse')

marks_data.fit(X_train1 , y_train,epochs = 500)

loss_df = pd.DataFrame(marks_data.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

marks_data.evaluate(X_test1,y_test)

X_n1 = [[30]]

X_n1_1 = Scaler.transform(X_n1)

marks_data.predict(X_n1_1)


```
## Dataset Information
![Screenshot 2024-02-25 020526](https://github.com/amurthavaahininagarajan/basic-nn-model/assets/118679102/ed18c24d-a0b1-46d0-b25d-ca2fa7b4adc0)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2024-02-25 020209](https://github.com/amurthavaahininagarajan/basic-nn-model/assets/118679102/9f058a72-f4f1-4e1a-9f7c-4f23ed075c77)


### Test Data Root Mean Squared Error

![Screenshot 2024-02-25 020238](https://github.com/amurthavaahininagarajan/basic-nn-model/assets/118679102/e1f55959-9364-4f74-abc6-89e292c71411)


### New Sample Data Prediction
![Screenshot 2024-02-25 020303](https://github.com/amurthavaahininagarajan/basic-nn-model/assets/118679102/d9682226-c3b2-4125-aef8-e542328b422e)


## RESULT
Thus a neural network regression model is developed for the created dataset.
