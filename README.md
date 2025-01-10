# Turkish-Sentiment-Analysis

This Project is a Machine Learning project. The purpose of the project is to determine whether the data received from the user is "Positive" or "Negative".

The project's "main.py" file is the main code file that performs the training, testing, and prediction processes of the model. It creates the "random_forest_model.pkl" and "vectorizer.pkl" files so that they can be used in the "interface.py" file.

"interface.py" is the Python file that creates the visual user interface of the application and interacts with the user.

"review_file.py" opens a "pkl" file and analyzes the type of content.

"comments.csv" includes 15.000 Turkish comments.

15% of the data is reserved for testing.

The "stopwords" folder is a folder that lists the stop words used in the project. These words are removed from the text during data preprocessing.

Binary ROC AUC Score: 0.9826323797183245

Confusion Matrix:
True Positive: 745
False Positive: 77
False Negative: 46
True Negative: 993
