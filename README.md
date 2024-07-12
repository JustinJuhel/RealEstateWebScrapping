# Real Estate Project

The goal of this project is to build a web scrapping tool to get all information I need to analyse geographic zones, markets and offers for real estate investment.

We can imagine the tool as having several scopes:
- Track offers with the filters I chose
	- Store in one or several files the main elements of a real estate offer. Not forget the images, the starting and ending dates of the offers (interesting to track how long the offers are available).
	- Scrap from different websites to group all offers in one file.
	- Further features...
- Carry out market researches in a defined geographical area
	- Get the average, min, max price, maybe a histogram to see the price distribution.
	- Get information about the surface, number of rooms of the housing sold in the area.
	- Get the rental prices for one month and also on Airbnb or Booking.
	- Further features...

We can imagine several steps in the project development:
- The highest added value is for the market researches, thus we can begin with this. The MVP just performs the action of demanding from the user a city (postal code and city name) and a range, and take all available data in the geographic area and put data into a CSV file.
- The second step consists in following the offers. Creating a CSV file for a given area and track every day the available offers in the area with all information from them. For example, the tool should be capable of tracking the price from an announcement to see when it gets modified it should have enough data to know how long the announcement stay online before finding a purchaser.
- One main problem to keep track of the announcements is that real estate agents often delete the announcement to republish the same one with different pictures, so that we think it is a new one. A further step is then to build a neural network capable of predicting if the images of a new offer correspond to the same flat as a previous offer, although they are different.
- Another step would be to build a website to manage the data easily.

Here is a part of a conversation with Mistral AI about how to solve the problem of the real estate agents re-posting the same flats with different images.

Me:

I have a practical problem. I try to keep track of real estate announcements on the web, but real estate agents often delete one offer to republish it with other pictures and a new price, instead of just modifying the announcement, so that we think it is a new announcement. I have a python tool that stores all offers and their caracteristiscs in a CSV file. I would like to improve my tool so that it recognizes when a new announcement correspond to a flat that it already has in its database. It does not have to be 100% sure that this new announcement is the same flat as this old anouncement, but it can emit a doubt and I can verify myself. What solution, what technology would you use to solve this problem?

Mistral:

To solve this problem, you can use a combination of text processing and machine learning techniques. Here's a possible solution:
1. **Text Processing**: Use Natural Language Processing (NLP) techniques to extract features from the text of the real estate announcements. This could include things like the number of rooms, location, size, and any other relevant information. You can use libraries like NLTK, SpaCy, or Gensim in Python for this.
2. **Feature Extraction**: Convert the textual data into numerical features that can be used for machine learning. This could involve techniques like TF-IDF, Word2Vec, or BERT embeddings.
3. **Similarity Measure**: Use a similarity measure to compare the features of the new announcement with the features of the announcements in your database. Cosine similarity is a common choice for this, but you could also use Euclidean distance or Jaccard similarity, depending on your specific needs.
4. **Machine Learning**: Train a machine learning model to classify whether a new announcement corresponds to a flat that is already in your database. You could use a supervised learning approach, where you label a subset of your data as "same" or "different", and then train a model (like a logistic regression, SVM, or random forest) to predict the labels of the unlabeled data.
5. **Doubt Emission**: If the model is not very confident about its prediction, you can emit a doubt and manually verify the announcement. You can do this by setting a threshold on the confidence score of the model's predictions.
6. **Update Mechanism**: If the model determines that a new announcement corresponds to a flat that is already in your database, you can update the existing entry in your database with the new information.

This solution combines the power of NLP and machine learning to solve your problem. It's not a perfect solution, but it should be able to significantly improve the accuracy of your tool.

Me:

Would you use a CNN to recognise if the new images correspond to the same flat as the previous images ?

Mistral:

Yes, you could potentially use a Convolutional Neural Network (CNN) to recognize if the new images correspond to the same flat as the previous images. However, it's important to note that this would be a more complex solution than the one I described earlier, and it would require a larger dataset to train the model.

Here's how you could use CNNs to solve this problem:
1. **Image Preprocessing**: Preprocess the images by resizing them to a fixed size, normalizing the pixel values, and augmenting the data with techniques like rotation, translation, and flipping.
2. **Feature Extraction**: Use a pre-trained CNN (like VGG16 or ResNet) to extract features from the images. You can use the output of the penultimate layer of the CNN as the feature vector for each image.
3. **Similarity Measure**: Use a similarity measure to compare the feature vectors of the new images with the feature vectors of the images in your database. Cosine similarity is a common choice for this, but you could also use Euclidean distance or other similarity measures.
4. **Machine Learning**: Train a machine learning model to classify whether a new set of images corresponds to a flat that is already in your database. You could use a supervised learning approach, where you label a subset of your data as "same" or "different", and then train a model (like a logistic regression, SVM, or random forest) to predict the labels of the unlabeled data.
5. **Doubt Emission**: If the model is not very confident about its prediction, you can emit a doubt and manually verify the images. You can do this by setting a threshold on the confidence score of the model's predictions.
6. **Update Mechanism**: If the model determines that a new set of images corresponds to a flat that is already in your database, you can update the existing entry in your database with the new information.

Using CNNs to solve this problem can potentially improve the accuracy of your tool, especially if the images contain important visual cues that are not captured by the text of the announcements. However, it's a more complex solution than the one I described earlier, and it would require a larger dataset to train the model.