# ML-Challenge
This code was used for the purposes of a group assignment (max 3 members) for the course in "Machine Learning" of the programme MSc Data Science at the University of Tilburg. The results of our predictions, and those of the other groups, were uploaded on Codalab. This assignment was our first attempt at ML and DL and it was submitted on **11/12/2019**, therefore it is reflective of our skills and knowledge at the time of submission.

# Intro
Our group participated in a speech classification challenge, using single-word audio files, extracted Mel-Frequency Cepstral Coefficients (MFCC) features and train labels to predict the word for each audio file in the test set. We ended up using a Convolutional Neural Network for the purposes of this challenge.                                       

# Data Description
Speech Classification

In this challenge the task is to learn to recognize which of several English words is pronounced in an audio recording. 
The following files were provided to us: 
1. feat.npy: anarray with Mel-frequency cepstral coeffcients extracted from each wav ﬁle. The features at index i in this array were extracted from the wav ﬁle at index i of the array in the ﬁle path.npy.
2. path.npy: an array with the order of wav ﬁles in the feat.npy array.
3. train.csv: this ﬁle contains two columns: path with the ﬁlename of the recording and word with word which was pronounced in the recording. This is the training portion of the data.
4. test.csv: This is the testing portion of the data, and it has the same format as the ﬁle train.csv except that the column word is absent.

Note - Due to the university's policy and regulations, the dataset cannot be made publicly available.

# Evaluation Metric
Classification Accuracy - My team achieved a test accuracy of 90.67%.
