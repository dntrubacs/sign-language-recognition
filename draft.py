# chekcing to see if we can load the model
from tensorflow import keras
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from ModelEvaluation import evaluate
sr_model = keras.models.load_model('sign_recongiton_model')

'''
Reading the data from a CSV file and transforming into a numpy array to feed into the neural net
'''
# reading the raw data
test_data = pd.read_csv('sign_mnist_test.csv')
train_data = pd.read_csv('sign_mnist_train.csv')

# transforming to numpy array
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# separating the label from the images
train_labels = train_data[:,0]
test_labels = test_data[:,0]

# dividing by 255 to have all elements in the tensor between 0 and 1
train_images = train_data[:,1:]/255
test_images = test_data[:,1:]/255

# reshaping the train images to 28x28 matrices (representing a image in gray scale)
train_images = np.reshape(train_images,(27455,28,28))
test_images = np.reshape(test_images,(7172,28,28))



'''
Tokenizing the data (chancing from scalars to vectors with binary elements) for faster and better training.
'''
# tokenizing the labels
def one_hot(a):
  one_hot = np.zeros((len(a),25))
  for i in range(len(a)):
    one_hot[i][a[i]] = 1
  return one_hot

test_labels = one_hot(test_labels)
train_labels = one_hot(train_labels)

# reshaping the train and test data (28 x 28 images) to have 1 collor chanel
train_images = np.reshape(train_images,(27455,28,28,1))
test_images = np.reshape(test_images,(7172,28,28,1))

# taking some validation data
val_images = test_images[6000:]
val_labels = test_labels[6000:]

# correcting the test data
test_images = test_images[:6000]
test_labels = test_labels[:6000]

#real labels
sign_labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y']

def max_index(a):
   for i in range(len(a)):
     if a[i] == max(a):
       return i
#def evaluate(model,image):
 #   image = np.reshape(image,(1,28,28,1))
  #  prediction = np.reshape(sr_model.predict(image),(25,))
   # return max_index(prediction)

for i in range(10):
  img = np.reshape(test_images[i],(28,28))
  plt.imshow(img,cmap='gray')
  plt.show()
  prediction = evaluate(sr_model,img)
  expected = max_index(test_labels[i])
  print("Simulation",i)
  print("Prediction",prediction)
  print("Reality",sign_labels[expected])
  print(" ")