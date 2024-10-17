## Natural Language Processing Approach to Sentiment Analysis and The Development of Multi-Class Models with Deployment
The project leveraged on the concept of natural language processing to perform sentiment analysis and classify tweets based on the emotions sentiments present in them. The project further investigate the impact of multiple preprocessing strategies for text and developed multiclass models stochastic models, deep learning models and fine-tuned a pretrained models from huggingface. Also, estimate the performance of each models and established the comparisons according to their overall performance and generalization.

### Project Structure and Packages:
![nlp11](https://github.com/user-attachments/assets/d956b5f6-d95d-48bc-9a76-aa7a3d3eae37)

### Strategies and Techniques:
![nlp22](https://github.com/user-attachments/assets/983e1214-4bae-466e-89f1-f8f516cfa540)
Train different classification models relying mainly on the following:
0. Stochastic Modeling Approach.
1. A Fully Connected Neural Network.
2. A Recurrent Neural Network, based on LSTM or GRU.
3. A fine-tuned Transformer Architecture from a pretrained model that can be found on sites like HuggingFace.
4. Compare the different models to find the best approach and explained what you have learnt from this exercise and how would you proceed with another text classification use case.

### Analyzing the most dominant emotions:
![nlp33](https://github.com/user-attachments/assets/42ec1841-bd50-4633-ba66-34d7006423a4)

### Various Strategies and Model Development with Evaluation:
##### Term Frequency-Inverse Document Frequency Vectorization
![nlp44](https://github.com/user-attachments/assets/5ec41dcb-9856-43d6-adfc-814b1b636619)
![nlp55](https://github.com/user-attachments/assets/4a8f25f5-5d2e-4125-a1c3-6dc7c533c2f0)
![nlp66](https://github.com/user-attachments/assets/e84998f8-1dae-4aa1-bb06-7c68c292fd46)
![nlp77](https://github.com/user-attachments/assets/e5e43d10-a750-4b31-b8d4-b0199edeef40)

We observed that the recall score for the model using the `TF-IDF` vectorization stood at 81.0% and the confusion matrix further shows the areas where the model is misclassifying labels. It shows that the instance of joy is misclassified as love, anger, surprise, fear, sadness. For better performance, the model need to be fine-tune and retrained by agrregating the labels and reclassify them to fewer categories.

##### Bag of Words (BoW) Vectorization using CountVectorizer
![nlp88](https://github.com/user-attachments/assets/d05a5dcf-b00f-477e-b0dd-a2775b329ece)
![nlp99](https://github.com/user-attachments/assets/6004f253-c799-4b99-bc38-d373fc96a759)
![nlp10](https://github.com/user-attachments/assets/dbfa3f99-b48f-4e62-9a18-457dfee8fa12)

We observed that the recall score for the model using the `BoW` vectorization stood at 86.0% and the confusion matrix further shows the areas where the model is misclassifying labels. It shows that the instance of joy was misclassified as love and viceversa. For better performance, the model need to be fine-tune and retrained by agrregating the labels and reclassify them to fewer categories.

### Tokenization with Tensorflow
Under the tokenization strategy, I developed different deep learning models and measures their performance.
#####  Classification model relying on a Fully Connected Neural Network (FCNN)
![nlp100](https://github.com/user-attachments/assets/d1563fb9-912e-4f79-8ecb-d0ed0331bfc2)
![nlp102a](https://github.com/user-attachments/assets/07a80c0d-b705-409c-95be-fe3a54b58737)
![nlp101](https://github.com/user-attachments/assets/8821bf20-dbf4-4fb7-a120-77ffe4ce737d)
There was overfitting using the fully connected neural network and I improved the algorithms in the subsequent methods.

##### Recurrent Neural Network using LSTM
![nlp102](https://github.com/user-attachments/assets/e6a5a449-ba5d-4bbd-8023-6550f88671e5)
![nlp103](https://github.com/user-attachments/assets/755362a3-e31f-41ef-b068-18237857a2f2)
There was correction in the model overfitting and the validation loss has been greatly minimized with this model and this outperformed the `FCNN`.

##### Recurrent Neural Network using GRU
![nlp104](https://github.com/user-attachments/assets/8089f0f7-7e87-4042-9c41-3f1ab89f0561)
![nlp105](https://github.com/user-attachments/assets/63d29c92-dc16-4772-a953-54cc931c543c)
![nlp106](https://github.com/user-attachments/assets/01794c0f-c9c5-419c-99d4-b3778fce4aae)
![nlp107](https://github.com/user-attachments/assets/0c538f24-96e0-45d5-845c-f89e34bb6fff)

Overall, the `LSTM` outperformed the `GRU` with an accuracy of 0.88.
I fine-tuned a Bidirectional Encoders for Representations Transformers `BERT` and a very small sample were used because of computing resources with the trends of improving the performance of the model.
![nlp108](https://github.com/user-attachments/assets/b26536ed-ed46-4d35-830c-4e24bfb98c5e)

###### Deployment of model and Industrialization
![nlp109](https://github.com/user-attachments/assets/2eace5b7-3854-4959-b7e6-f9e9c60e75a9)

### Conclusion
After comparing the results, I obsereved that RNN with LSTM_Improved performs better on all the sentiments classes. The accuracy of the LSTM_IMPROVED was 89.0% which is only measuring the probability of true predictions with its precison and recall averagely at 89.0%

This was validated with the LSTM_IMPROVED precision weighted average of 0.89, same as F1-score and recall. The essence of considering these metrics is to to ascertain that the model was able to identify and makes accurate classification for any sentiments present in any given sentence or text.

Key takeaways from the comparison may include:
Some models may perform better on certain sentiment classes in the case of FCNN, LSTM and GRU.

The fine-tuned BERT model has the potential of outperforming all other model as it was observed of it relative increase accuracy on a very small amount of data. The choice of the small amount of text sample taken was due to computational and resources available at the time of this analysis

They were Overfitting problems experienced by some of the models but this could be corrected by hyperparametrizing and further tuning of all existing models.

With 10 Epochs set for each models, The training time and resource requirements were less in FCNN, LSTM and GRU compared to epochs = 3 for BERT pretrained models and small amount of samples still takes very longer time to compute.

Based on these piece of work and observations, I recommend that the best approach is to optimize neural networks with memory term algorithms and fine-tune the parameters to obatin a more robust results especially on classification analysis for text.

###### Contribution:
I welcome contributions to this project! Here's how you can contribute:

Fork the Repository
Reach out for ways to improve on future works
Clone the Repository
Thank you for considering contributing to this project!

Author:
Name - Olanrewaju Adegoke

Email - Larrysman2004@yahoo.com








