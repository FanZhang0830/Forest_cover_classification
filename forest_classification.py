# Section #1 import tools I need
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.compose import ColumnTransformer

# Section #2 input and visulize dataframe
cover_data = pd.read_csv('https://raw.githubusercontent.com/FanZhang0830/Forest_cover_classification/main/cover_data.csv?token=GHSAT0AAAAAABRKZF4QXJVOVRRSC3EKT7SWYQPYLOQ')
print(cover_data.head())
print(cover_data['Elevation'].describe())

# numerical data visualization
# plt.figure(figsize=(9, 8))
# sns.distplot(cover_data['Elevation'], color = 'g',  bins=100, hist_kws={'alpha': 0.4})
# plt.show()
# use pandas visualiation tool to visulize numerical data
cover_data.hist(figsize=(16,5), bins = 50, xlabelsize = 8, ylabelsize = 8)
plt.show()

# Section #3 preprocessing cover_data for neural network model
labels = cover_data.iloc[:, -1]
features = cover_data.iloc[:, 0:-1]
num_features = features.shape[1]

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state= 42)

numerical_columns_ct = ['Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

# scaled train and test datasets
ct = ColumnTransformer([('normalize', StandardScaler(), numerical_columns_ct)],remainder='passthrough')
features_train_scaled = ct.fit_transform(features_train)
features_test_scaled = ct.transform(features_test)
# 


# Section #4 design model
def model_design(features):
  model = tf.keras.Sequential()
  input_layer = tf.keras.layers.InputLayer(input_shape = (num_features,))
  model.add(input_layer)
  # add the first hidden layers
  model.add(tf.keras.layers.Dense(128, activation = 'relu'))
  # add the second hidden layers
  model.add(tf.keras.layers.Dense(64, activation = 'relu'))
  # add output layer
  model.add(tf.keras.layers.Dense(1, activation = 'softmax'))
  return model

# Section #5 model training and evaluating
my_model = model_design(features_train)

# model optimizer design
opt = tf.keras.optimizers.Adam(learning_rate = 0.01)

# model compiling for a multi-class classification model
my_model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

my_model.fit(features_train, labels_train, epochs = 2, batch_size = 2, verbose = 1)

mae, mse = my_model.evaluate(features_test, labels_test, verbose = 0)
print('mae = {}'.format(mae))

my_model.fit(features_train, labels_train, epochs=2, batch_size=3, verbose=1, validation_split = 0.2)

# test model
my_model.summary()
predictions = my_model.predict(features_test)
print('test1 = {}'.format(predictions[1]))