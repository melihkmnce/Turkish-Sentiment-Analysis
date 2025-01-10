# Turkish-Sentiment-Analysis

The project's "main.py" file is the main code file that performs the training, testing, and prediction processes of the model. It creates the "random_forest_model.pkl" and "vectorizer.pkl" files so that they can be used in the "interface.py" file.

"interface.py" is the Python file that creates the visual user interface of the application and interacts with the user.

"comments.csv" includes 15.000 Turkish comments.

The "stopwords" folder is a folder that lists the stop words used in the project. These words are removed from the text during data preprocessing.

Binary ROC AUC Score: 0.9826323797183245

Confusion Matrix:
[[746,77],
 [46,993]]
