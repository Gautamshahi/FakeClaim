import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
from  IPython import display

import pathlib
import shutil
import tempfile
import string
import gc

import numpy as np 
import pandas as pd 

import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

import tensorflow_hub as hub
import tensorflow_datasets as tfds

## Install additional packages
!pip install -q git+https://github.com/tensorflow/docs

import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

import keras.backend as K

import os
nltk.download('stopwords')

# processed data - comments from multiple sources with their corresponding labels
df = pd.read_csv('../input/fake-news-youtube/processed_ycomments_vtitle_ntext.csv')

if not (df.index.is_monotonic_increasing and df.index.is_unique):
    df.reset_index(inplace=True, drop=True)

df["comment_wvt_tt"] = df["comment_wvt_tt"].replace({np.nan: "undefined"})

CLASS_NAMES = ['fake', 'real']
class_mapper = {
    'fake':0,
    'real':1
}

df['target_label'] = df['target_label'].map(class_mapper)
print(df.shape)

stop_words = stopwords.words('english')
def text_preprocessing(text):
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    pure_text = ' '.join(filtered_words)
    pure_text = pure_text.translate(str.maketrans('', '', string.punctuation)).strip()
    return pure_text


X_train, X_test, y_train, y_test = train_test_split(df.comment_wvt_tt, df.target_label, test_size=0.3, random_state=42, stratify = df.target_label.values)

X = df.comment_wvt_tt.apply(text_preprocessing).to_numpy()
y = df.target_label.to_numpy().astype('float32').reshape(-1, 1)

train_X, test_X, train_y, test_y = train_test_split(X, y,
                                                 train_size=0.84,
                                                 stratify=y,
                                                 random_state=42)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,
                                                 train_size=0.84,
                                                 stratify=train_y,
                                                 random_state=42)



model_name = "BERTFakeNewsDetector"
model_callbacks = ModelCheckpoint(model_name, save_best_only=True)

bert_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(bert_name,
                                         padding='max_length',
                                         do_lower_case=True,
                                         add_special_tokens=True)


def tokenize(df):
    inputs = tokenizer(df.tolist(),
                      padding=True,
                      truncation=True,
                      return_tensors='tf').input_ids
    return inputs


train_X_encoded = tokenize(train_X)
val_X_encoded = tokenize(val_X)
test_X_encoded = tokenize(test_X)

def prepare_datasets(encoded, true_df, true_target_df):
    return tf.data.Dataset.from_tensor_slices((encoded, true_target_df)).shuffle(true_df.shape[0]).batch(8).prefetch(tf.data.AUTOTUNE)

train_ds = prepare_datasets(train_X_encoded, train_X, train_y)
test_ds = prepare_datasets(test_X_encoded, test_X, test_y)
val_ds = prepare_datasets(val_X_encoded, val_X, val_y)

## Setting-up Models
model = TFAutoModelForSequenceClassification.from_pretrained(bert_name, num_labels=2)

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

model.compile(
        optimizer = Adam(learning_rate=1e-5),
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name='Accuracy'),
            #K.eval(f1),
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall')
        ]
    )
model_history = model.fit(train_ds,
                     validation_data=val_ds,
                     callbacks=model_callbacks,
                     epochs=5,
                     batch_size=16)

model_history = pd.DataFrame(model_history.history)

## Main models

def train_and_evaluate_model(module_url, embed_size, name, trainable=False):
    hub_layer = hub.KerasLayer(module_url, input_shape =[], output_shape =[embed_size], dtype = tf.string, trainable =trainable)
    model = tf.keras.models.Sequential([
      hub_layer,
      tf.keras.layers.Dense(256, activation = 'relu'),
      tf.keras.layers.Dense(64, activation ='relu'),
      tf.keras.layers.Dense(1, activation ='sigmoid')
  ])
    model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss= tf.losses.BinaryCrossentropy(),
                metrics=[
            tf.keras.metrics.Precision(name='Precision'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.metrics.AUC(curve='ROC',name='AUC')])
    model.summary()
    history = model.fit(X_train,y_train,
                      epochs =5,
                      batch_size=8,
                      validation_data = (X_test,y_test),
                      callbacks = [tfdocs.modeling.EpochDots(),
                                   tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min'),
                                   tf.keras.callbacks.TensorBoard(logdir/name)],
                      verbose=0)
    return history

histories = {}
logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

# Model 1
module_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" 
histories['gnews-swivel-20dim'] = train_and_evaluate_model(module_url, embed_size=20, name='gnews-swivel-20dim', trainable=False)

# fine tuning
module_url = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1" 
histories['gnews-swivel-20dim-finetuned'] = train_and_evaluate_model(module_url, embed_size=20, name='gnews-swivel-20dim-finetuned', trainable=True)

gc.collect()

plt.rcParams['figure.figsize'] = (12, 8)
plotter = tfdocs.plots.HistoryPlotter(metric = 'AUC')
plotter = tfdocs.plots.HistoryPlotter(metric = 'Precision')
plotter = tfdocs.plots.HistoryPlotter(metric = 'Recall')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Models Result")
plt.show()

# Model 2
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
histories['universal-sentence-encoder'] = train_and_evaluate_model(module_url, embed_size=512, name='universal-sentence-encoder', trainable=False)

plt.rcParams['figure.figsize'] = (12, 8)
plotter = tfdocs.plots.HistoryPlotter(metric = 'AUC')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("AUC Curves for Models")
plt.show()

plotter = tfdocs.plots.HistoryPlotter(metric = 'loss')
plotter.plot(histories)
plt.xlabel("Epochs")
plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
plt.title("Loss Curves for Models")
plt.show()





