import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('../data.pickle', 'rb'))

data = np.asarray(data_dict['data'])  # Features
labels = np.asarray(data_dict['labels'])  # Corresponding labels

label_encoder = LabelEncoder()  # Convert class labels to integers
integer_labels = label_encoder.fit_transform(labels)

onehot_encoder = OneHotEncoder(sparse=False)
labels_onehot = onehot_encoder.fit_transform(integer_labels.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(
    data, labels_onehot, test_size=0.2, shuffle=True, stratify=labels
)

model = Sequential([
    Dense(42, input_shape=(data.shape[1],), activation='relu'),  # First layer
    Dropout(0.5),  # Dropout for regularization
    Dense(64, activation='relu'),  # First hidden layer
    Dropout(0.5),
    Dense(32, activation='relu'),  # Second hidden layer
    Dense(labels_onehot.shape[1], activation='softmax')  # Output layer
])

model.compile(
    optimizer=Adam(learning_rate=0.001),  # Optimizer
    loss='categorical_crossentropy',  # Loss function for multi-class classification
    metrics=['accuracy']  # Metric to track during training
)

history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=50,  # Number of iterations over the data
    batch_size=32,  # Number of samples per training batch
    verbose=1  # Show progress during training
)

y_test_pred = model.predict(x_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_actual_classes = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_actual_classes, y_test_pred_classes)
print('{}% of samples were classified correctly!'.format(accuracy * 100))

model.save('sign_language_model.h5')  # Save the model in HDF5 format

with open('label_encoders.p', 'wb') as f:
    pickle.dump({'label_encoder': label_encoder, 'onehot_encoder': onehot_encoder}, f)
