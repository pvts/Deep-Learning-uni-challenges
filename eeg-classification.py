{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 179
    },
    "colab_type": "code",
    "id": "kfoUcWEVp2_E",
    "outputId": "41cf49ef-41fd-4e89-da76-2e2da1486b4f"
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "#importing the necessary libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import tensorflow as tf\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import precision_score, recall_score, f1_score, classification_report\n",
    "from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D, \\\n",
    "BatchNormalization, SpatialDropout1D\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.utils import to_categorical\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint\n",
    "from imblearn.combine import SMOTETomek\n",
    "import warnings\n",
    "#Surpressing all deprecation and future warnings\n",
    "warnings.filterwarnings(\"ignore\", category=DeprecationWarning)\n",
    "warnings.filterwarnings(\"ignore\", category=FutureWarning)\n",
    "#Sets the global random tf seed\n",
    "tf.random.set_random_seed(42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 131
    },
    "colab_type": "code",
    "id": "3Cn9D6KmVCv8",
    "outputId": "889f2fc1-e7d0-4864-c4fb-a7d165922d0e"
   },
   "outputs": [],
   "source": [
    "#mounting the drive to this notebook\n",
    "from google.colab import drive\n",
    "drive.mount('/content/drive', force_remount=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Y9NjPM86XeMV"
   },
   "outputs": [],
   "source": [
    "#loading the dataset\n",
    "raw = np.load('/content/drive/My Drive/data/Data_Raw_signals.pkl', allow_pickle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "Z9bdrbimlXiH"
   },
   "outputs": [],
   "source": [
    "#Splitting our features and targets\n",
    "feat = raw[0]\n",
    "targets = raw[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "colab_type": "code",
    "id": "dw1R5Tjklrqs",
    "outputId": "3d8281ca-eb76-4aa6-a3ab-43db639dcd43"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "#Checking for missing data in the dataset - no NAs\n",
    "print(np.isnan(feat).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 265
    },
    "colab_type": "code",
    "id": "GzORg2ROWjkI",
    "outputId": "3631eeed-b8e6-4720-afba-642a08f8d367"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "tags": []
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "#vizualizing the signals\n",
    "signal = feat.reshape(feat.shape[0], feat.shape[1] * feat.shape[2])\n",
    "plt.plot(signal, 'r')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "9lq20mXyyfuT"
   },
   "outputs": [],
   "source": [
    "#Reshaping the features to convert them to 2D, to feed it into SMOETomek \n",
    "feat_X = np.reshape(feat, (feat.shape[0], feat.shape[1] * feat.shape[2]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "7Cj0J9MOxBkq"
   },
   "outputs": [],
   "source": [
    "#OverSampling the data using SMOTETomek, since we have an imbalanced dataset\n",
    "smk = SMOTETomek(random_state=42)\n",
    "feat_X, targets_Y = smk.fit_resample(feat_X, targets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 73
    },
    "colab_type": "code",
    "id": "FNgEmfPEmO-7",
    "outputId": "423c153e-152c-4b22-f5c2-d79820ff6586"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial Shape: (15375, 6000) \n",
      "Dimensions: 2\n",
      "Final shape: (15375, 3000, 2)\n"
     ]
    }
   ],
   "source": [
    "#Shape and dimensions of the dataset after the transformation\n",
    "print(f\"Initial Shape: {feat_X.shape} \\nDimensions: {feat_X.ndim}\")\n",
    "\n",
    "#Reshaping it to get to the desired shape of samples, timesteps, features\n",
    "feat_x = feat_X.reshape(feat_X.shape[0], feat.shape[2], feat.shape[1])\n",
    "print('Final shape:',feat_x.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "uruuw83Lzhwz"
   },
   "outputs": [],
   "source": [
    "#Setting the X and y and subsequently splitting the data\n",
    "X = feat_x\n",
    "y = targets_Y\n",
    "\n",
    "#test size of 10% with random_state of 62, stratifying and shuffling the data\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, shuffle=True,\n",
    "                                                    stratify=y, random_state=62)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 54
    },
    "colab_type": "code",
    "id": "ywk87pzWg1fe",
    "outputId": "6a1ade6f-cb57-418d-ed10-dba5ba62fd27"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 3 4 5] \n",
      "Number of labels: 6\n"
     ]
    }
   ],
   "source": [
    "#Encoding the labels\n",
    "lbl = LabelEncoder()\n",
    "y_train = lbl.fit_transform(y_train)\n",
    "y_test = lbl.transform(y_test)\n",
    "\n",
    "#We have 6 labels\n",
    "print(np.unique(y_test), f'\\nNumber of labels: {len(np.unique(y_test))}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 54
    },
    "colab_type": "code",
    "id": "SkVkGjKGjk80",
    "outputId": "f400156b-53dd-46e0-d9d3-e1fabca340fb"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "X_train shape: (13837, 3000, 2) \n",
      "X_test shape: (1538, 3000, 2)\n"
     ]
    }
   ],
   "source": [
    "#Original shape of our train and test set\n",
    "print(\"X_train shape:\",X_train.shape, \"\\nX_test shape:\",X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "LHzNafAuHHB2"
   },
   "outputs": [],
   "source": [
    "#encoding the test_data to 0s and 1s\n",
    "labels = 6\n",
    "\n",
    "y_test_cat = to_categorical(y_test, labels)\n",
    "y_train_cat = to_categorical(y_train, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 54
    },
    "colab_type": "code",
    "id": "5MMQDKmpgsEj",
    "outputId": "8d380084-ad11-44f8-fc27-fba7ce1ecdc8"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "y_train_cat shape: (13837, 6)\n",
      "y_test_cat shape: (1538, 6)\n"
     ]
    }
   ],
   "source": [
    "#New shape\n",
    "print(\"y_train_cat shape:\",y_train_cat.shape)\n",
    "print(\"y_test_cat shape:\",y_test_cat.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "XEzIybKM8t1v"
   },
   "outputs": [],
   "source": [
    "#defining checkpoint and the early stopping to prevent overfitting\n",
    "checkpoint = ModelCheckpoint(filepath='bestmodel.hdf5', monitor='val_loss', \n",
    "                             save_weights_only=True,  mode='min', save_best_only=True, verbose=1)\n",
    "Early_Stopping = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "jQSB_614G9pW"
   },
   "outputs": [],
   "source": [
    "#Defining callbacks and our validation set\n",
    "callbacks = [Early_Stopping, checkpoint]\n",
    "validation = [X_test, y_test_cat]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "colab_type": "code",
    "id": "tllWmg86dxIm",
    "outputId": "4859c384-7843-478e-c55b-3658ff015845"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(3000, 2)\n"
     ]
    }
   ],
   "source": [
    "#input shape to feed to our CNN\n",
    "final_shape = X_train[0].shape\n",
    "print(final_shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "F0jyaikuRiTu"
   },
   "outputs": [],
   "source": [
    "#Conv1D Model\n",
    "model = Sequential()\n",
    "model.add(Conv1D(filters=128, kernel_size=(60), strides=2, padding='same',\n",
    "                 activation='relu', input_shape=(final_shape)))\n",
    "model.add(MaxPooling1D(12))\n",
    "model.add(BatchNormalization())\n",
    "model.add(SpatialDropout1D(rate=0.2))\n",
    "\n",
    "model.add(Conv1D(filters=108, kernel_size=(60), strides=2, activation='relu', padding='same'))\n",
    "model.add(MaxPooling1D(9))\n",
    "model.add(BatchNormalization())\n",
    "model.add(SpatialDropout1D(rate=0.2))\n",
    "\n",
    "model.add(Conv1D(filters=86, kernel_size=(60), strides=2, padding='same', activation='relu'))\n",
    "model.add(MaxPooling1D(4))\n",
    "model.add(BatchNormalization())\n",
    "model.add(SpatialDropout1D(rate=0.2))\n",
    "model.add(Flatten())\n",
    "\n",
    "model.add(Dense(68, activation='relu'))\n",
    "model.add(BatchNormalization())\n",
    "model.add(Dropout(0.3))\n",
    "\n",
    "model.add(Dense(6, activation='softmax'))\n",
    "\n",
    "opt='adagrad'\n",
    "#Compiling the model\n",
    "model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])\n",
    "\n",
    "history = model.fit(X_train, y_train_cat, epochs=100, batch_size=1054, verbose=1, \n",
    "                        validation_data=validation, callbacks=callbacks)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "zbCXNk51VKbp"
   },
   "outputs": [],
   "source": [
    "#plotting the model diagram\n",
    "tf.keras.utils.plot_model(\n",
    "    model, to_file='model.png', show_shapes=False, show_layer_names=True,\n",
    "    rankdir='TB', expand_nested=False, dpi=96\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "sECuRru6zYWJ"
   },
   "outputs": [],
   "source": [
    "#Load the best weights\n",
    "model.load_weights('bestmodel.hdf5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 54
    },
    "colab_type": "code",
    "id": "lrH_qwvWTgz3",
    "outputId": "f733982d-1c07-495c-c1ef-57645a2a3b7b"
   },
   "outputs": [],
   "source": [
    "#Conv1D -> calcuting validation accuracy and loss\n",
    "val_loss, val_acc = model.evaluate(X_test, y_test_cat, verbose=0)\n",
    "print('Validation loss:',val_loss, '\\nValidation accuracy:',val_acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "WhnDwmSngqaE"
   },
   "outputs": [],
   "source": [
    "#overview of the model\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "0xfjjCBpg0kn"
   },
   "outputs": [],
   "source": [
    "y_pred = model.predict_classes(X_test, labels)\n",
    "y_pred = to_categorical(y_pred, labels)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "DRDW-eM0i2Ww"
   },
   "outputs": [],
   "source": [
    "#Classification report\n",
    "print(classification_report(y_test_cat, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "kQrlJzs0g3TD"
   },
   "outputs": [],
   "source": [
    "#As above, showing the results of precision, recall and f1-score\n",
    "print(\"Precision:\", round(precision_score(y_test_cat, y_pred, average = 'macro') * 100, 2))\n",
    "print(\"Recall:   \", round(recall_score(y_test_cat, y_pred, average = 'macro') * 100, 2))\n",
    "print(\"f1-score: \", round(f1_score(y_test_cat, y_pred, average = 'macro') * 100, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "colab_type": "code",
    "id": "ZZAtGZXwDeQV",
    "outputId": "a028a1c8-e415-48f6-a092-0e8c72b12588"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])"
      ]
     },
     "execution_count": 96,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#checking the keys of the dictionary\n",
    "history.history.keys()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "hUTQo4q4Dgv3"
   },
   "outputs": [],
   "source": [
    "#setting the names before plotting\n",
    "accuracy = history.history['acc']\n",
    "val_acc = history.history['val_acc']\n",
    "loss = history.history['loss']\n",
    "val_loss = history.history['val_loss']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 295
    },
    "colab_type": "code",
    "id": "3OHPtqz9mJ-t",
    "outputId": "442cffde-a44b-463e-e512-80c1d52e05f9"
   },
   "outputs": [],
   "source": [
    "#Plotting train vs validation accuracy\n",
    "plt.plot(accuracy,'r')\n",
    "plt.plot(val_acc, 'b')\n",
    "plt.title('Model Accuracy')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.xlabel('Epochs')\n",
    "plt.legend(['Training accuracy', 'Validation Accuracy'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 295
    },
    "colab_type": "code",
    "id": "kOx6r3P_mL-s",
    "outputId": "08ca4e26-8988-4168-a4f7-76185770f8be"
   },
   "outputs": [],
   "source": [
    "#Plotting train vs validation loss\n",
    "plt.plot(loss, 'r')\n",
    "plt.plot(val_loss, 'b')\n",
    "plt.title('Model Loss')\n",
    "plt.ylabel('Loss')\n",
    "plt.xlabel('Epochs')\n",
    "plt.legend(['Training Loss', 'Validation Loss'], loc='upper right')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "KlD2JNrMDPS0"
   },
   "outputs": [],
   "source": [
    "#Loading the test set\n",
    "test = np.load('/content/drive/My Drive/data/Test_Raw_signals_no_labels.pkl', allow_pickle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "colab_type": "code",
    "id": "HfsAKG49z7Ga",
    "outputId": "f4333ccc-eb28-4b44-b7c0-f25cb223fd45"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1754, 2, 3000)"
      ]
     },
     "execution_count": 101,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data = test[0]\n",
    "test_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "pLmGlcKQ0MM4"
   },
   "outputs": [],
   "source": [
    "#Saving the shape of the test set to be later used when reshaping it\n",
    "X_t_shape = test_data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 36
    },
    "colab_type": "code",
    "id": "H2jjCvteHuvT",
    "outputId": "564b360e-4f97-44e5-8c57-efab348ddb96"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1754, 2, 3000)"
      ]
     },
     "execution_count": 103,
     "metadata": {
      "tags": []
     },
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_t_shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "2fod8ka7hXKe"
   },
   "outputs": [],
   "source": [
    "#Reshaping the test data\n",
    "test_data_tf = test_data.reshape(X_t_shape[0], X_t_shape[2], X_t_shape[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "pCCT2-3L1p_3"
   },
   "outputs": [],
   "source": [
    "y_t = model.predict_classes(test_data_tf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "8hj3M34z1-B4"
   },
   "outputs": [],
   "source": [
    "y_test_decoded = lbl.inverse_transform(y_t)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 0,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "3LU9c2Ir2_TW"
   },
   "outputs": [],
   "source": [
    "#Save the text with the predicted labels\n",
    "np.savetxt('answer.txt', y_test_decoded, delimiter=',', fmt='%d')"
   ]
  }
 ],
 "metadata": {
  "accelerator": "GPU",
  "colab": {
   "collapsed_sections": [],
   "machine_shape": "hm",
   "name": " 1 - Final_Code.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
