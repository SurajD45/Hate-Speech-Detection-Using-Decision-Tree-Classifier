Hate Speech Detection Using Decision Tree Classifier


Project Description
In this project, we built a model to detect hate speech, offensive language, and non-offensive language in tweets. Using a labeled dataset of tweets, we trained a Decision Tree Classifier to categorize each tweet based on its content. The project involved understanding the data, cleaning it up, extracting useful features, and then building and evaluating the model. The goal was to create a tool that can automatically identify harmful or inappropriate language on social media.

Key Steps
1. Understanding the Data
We started by loading the dataset and exploring it to get a sense of what we were working with. We looked at how the data was distributed and checked for any missing values or anomalies.
2. Cleaning the Data
The text in the tweets was cleaned to remove any unnecessary information like URLs, special characters, and numbers. We also got rid of common stop words (like "the" or "and") and reduced words to their base forms (e.g., "running" to "run") to make the data easier to work with.
3. Extracting Features
We converted the cleaned tweets into numerical data that the machine learning model could understand using a technique called CountVectorizer. This step transformed the text into a format suitable for analysis.
4. Building the Model
We split the data into training and testing sets to see how well our model could learn from the data and then make predictions on new data. We used a Decision Tree Classifier for this task, which is a simple yet powerful machine learning algorithm.
5. Evaluating the Model
We checked how well the model performed by comparing its predictions to the actual labels. We found that our model correctly identified the type of language in the tweets 87% of the time. We also used a confusion matrix to see where the model was making mistakes.
Findings
Accuracy: The model achieved an accuracy of 87%, which is quite promising. It means the model correctly classified the tweets in 87 out of 100 cases.
Performance Insights: The model is good at identifying non-offensive language but sometimes confuses hate speech with offensive language. This suggests there might be some overlap between these categories in the dataset.
Next Steps
Improve the Model: We can try tuning the model's settings or even testing more advanced models like Random Forests or Gradient Boosting to see if we can get better accuracy.
Add More Features: Introducing additional features like TF-IDF or word embeddings might help the model better understand the context of the tweets.
Expand the Dataset: We could improve the model's performance by adding more data or using techniques to create more diverse training examples.
Cross-Validation: To ensure our model is robust, we could use cross-validation, which helps avoid overfitting and ensures the model generalizes well to new data.

Conclusion

This project successfully created a tool to detect hate speech and offensive language on social media, with an accuracy of 87%. While the model performs well, there's always room for improvement, especially in distinguishing between similar categories like hate speech and offensive language. Future work will focus on enhancing the model's accuracy and expanding its capabilities.



Installation and Usage Clone this repository.

Ensure you have Python and the required libraries installed.

Run the notebook or script to train the model and make predictions.
Requirements

Python 3.x

Pandas

NumPy

Scikit-learn

NLTK

Seaborn

Matplotlib
