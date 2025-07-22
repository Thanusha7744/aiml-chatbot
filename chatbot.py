from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

# FAQ data (30 Q&A)
faq_pairs = [
    ("What is AI?", "Artificial Intelligence is the simulation of human intelligence in machines."),
    ("What is machine learning?", "Machine learning is a subset of AI that allows systems to learn from data."),
    ("What is deep learning?", "Deep learning is a subset of ML that uses neural networks with many layers."),
    ("What is NLP?", "Natural Language Processing enables machines to understand and process human language."),
    ("What is underfitting?", "Underfitting happens when a model is too simple to capture patterns in the data."),
    ("What is overfitting?", "Overfitting happens when a model learns the training data too well, including noise."),
    ("What is supervised learning?", "Supervised learning uses labeled data to train models."),
    ("What is unsupervised learning?", "Unsupervised learning finds patterns in unlabeled data."),
    ("What is reinforcement learning?", "Reinforcement learning trains agents through rewards and penalties."),
    ("What is a neural network?", "A neural network is a series of algorithms mimicking the human brain."),
    ("What is a decision tree?", "A decision tree is a flowchart-like structure used for classification and regression."),
    ("What is a random forest?", "Random forest is an ensemble of decision trees for better accuracy."),
    ("What is a support vector machine?", "SVM is a supervised algorithm for classification tasks."),
    ("What is a confusion matrix?", "A confusion matrix shows predicted vs actual classifications."),
    ("What is precision?", "Precision is the ratio of correctly predicted positive observations to total predicted positives."),
    ("What is recall?", "Recall is the ratio of correctly predicted positives to all actual positives."),
    ("What is F1-score?", "F1-score is the harmonic mean of precision and recall."),
    ("What is a ROC curve?", "ROC curve shows the performance of a classification model at all thresholds."),
    ("What is gradient descent?", "Gradient descent is an optimization algorithm to minimize a cost function."),
    ("What is a cost function?", "Cost function measures the error between predicted and actual values."),
    ("What is feature engineering?", "Feature engineering is creating new input features to improve model performance."),
    ("What is dimensionality reduction?", "Dimensionality reduction reduces the number of input variables."),
    ("What is PCA?", "Principal Component Analysis is a technique for dimensionality reduction."),
    ("What is clustering?", "Clustering groups similar data points without labels."),
    ("What is K-means?", "K-means is a clustering algorithm to partition data into K clusters."),
    ("What is a hyperparameter?", "A hyperparameter is a configuration set before training the model."),
    ("What is cross-validation?", "Cross-validation is a technique to evaluate model performance on unseen data."),
    ("What is bias in ML?", "Bias is error introduced by approximating real-world problems with simple models."),
    ("What is variance in ML?", "Variance is error from sensitivity to small fluctuations in the training set."),
    ("What is an epoch?", "An epoch is one complete pass through the entire training dataset.")
]

questions = [q for q, _ in faq_pairs]
answers = [a for _, a in faq_pairs]

# Create TF-IDF model
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def get_response(user_input):
    # Correct spelling
    corrected_input = str(TextBlob(user_input).correct())

    # Transform and compute similarity
    user_tfidf = vectorizer.transform([corrected_input])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    max_index = cosine_similarities.argmax()
    max_score = cosine_similarities[0, max_index]

    print(f"DEBUG: Corrected input: {corrected_input}, Max score: {max_score}")

    # If confidence is too low
    if max_score < 0.5:  # stricter threshold
        return "🤖 Sorry, I don't understand your question. Please try rephrasing."

    return answers[max_index]