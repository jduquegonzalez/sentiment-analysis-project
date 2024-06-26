from string import punctuation, digits
import numpy as np
import random



#==============================================================================
#===  PART I  =================================================================
#==============================================================================



def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices



def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        `feature_vector` - numpy array describing the given data point.
        `label` - float, the correct classification of the data
            point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - float representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given data point and
        parameters.
    """
    # Your code here

    # Calculate the product of theta transpose and feature_vector plus theta_0
    prediction = np.dot(theta, feature_vector) + theta_0
    
    # Calculate the hinge loss
    hinge_loss = max(0, 1 - label * prediction)
    
    return hinge_loss

    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the hinge loss for given classification parameters averaged over a
    given dataset

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.
    Returns:
        the hinge loss, as a float, associated with the given dataset and
        parameters.  This number should be the average hinge loss across all of
    """

    # Initialize total loss
    total_loss = 0.0
    
    # Loop through each data point
    for i in range(feature_matrix.shape[0]):
        # Calculate hinge loss for the current data point
        loss = hinge_loss_single(feature_matrix[i], labels[i], theta, theta_0)
        # Add the loss to total loss
        total_loss += loss
    
    # Calculate the average hinge loss
    average_loss = total_loss / feature_matrix.shape[0]
    
    return average_loss

    raise NotImplementedError


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the perceptron algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.
    Returns a tuple containing two values:
        the updated feature-coefficient parameter `theta` as a numpy array
        the updated offset parameter `theta_0` as a floating point number
    """
    # Calculate the prediction
    prediction = np.dot(current_theta, feature_vector) + current_theta_0

    # Check if the prediction is incorrect (label * prediction <= 0)
    if label * prediction <= 0:
        # Update theta and theta_0
        current_theta = current_theta + label * feature_vector
        current_theta_0 = current_theta_0 + label
    
    return current_theta, current_theta_0
    raise NotImplementedError
"""
Initialization:

- Explains that the function implements the Perceptron algorithm for binary classification.
- Details the input parameters: training data (feature_matrix), labels (labels), and maximum iterations (T).
- Mentions initializing the weight vector (theta) and bias (theta_0) with zeros.

Training loop:

- Describes the loop that iterates for a maximum number of times (T).

Iterating over data points:

- Explains the inner loop that iterates through each data point in the training data (feature_matrix).

Calculate prediction:

- Comments that the prediction is calculated using the dot product of the weight vector (theta) and the current feature vector (feature_matrix[i]), then adding the bias (theta_0).

Check for incorrect prediction:

- Explains the condition that checks if the prediction has the wrong sign compared to the label (labels[i]). This indicates an incorrect prediction.

Update weights and bias:

- Comments that if the prediction is incorrect, the weights and bias are updated according to the Perceptron update rule.
- Explains that the update increases the weight for the feature in the current data point (feature_matrix[i]) that aligns with the correct label (labels[i]).
- Mentions that the bias is also updated to move the decision boundary in the direction of the correct classification.

Return weights:

- After the training loop, the function returns the learned weight vector (theta) and the bias (theta_0).
"""


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set: we do not stop early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the feature-coefficient parameter `theta` as a numpy array
            (found after T iterations through the feature matrix)
        the offset parameter `theta_0` as a floating point number
            (found also after T iterations through the feature matrix).
    """
 
    # Initialise theta and theta_0
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)

    return theta, theta_0

    raise NotImplementedError
"""
Explanation:
Initialize theta and theta_0:

Set theta to a zero vector of the same length as the number of features.
Set theta_0 to 0.0.
Iterate Through the Data:

Loop through the dataset T times.
For each iteration, get the order of samples using get_order(nsamples).
Update theta and theta_0 using perceptron_single_step_update for each data point.
Return the Final Parameters:

After completing T iterations, return the updated theta and theta_0.
"""



def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given dataset.  Runs `T`
    iterations through the dataset (we do not stop early) and therefore
    averages over `T` many parameter values.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: It is more difficult to keep a running average than to sum and
    divide.

    Args:
        `feature_matrix` -  A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns a tuple containing two values:
        the average feature-coefficient parameter `theta` as a numpy array
            (averaged over T iterations through the feature matrix)
        the average offset parameter `theta_0` as a floating point number
            (averaged also over T iterations through the feature matrix).
    """
    # Initialise theta and theta_0
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    theta_sum = np.zeros(feature_matrix.shape[1])
    theta_0_sum = 0.0
    nsamples = feature_matrix.shape[0]

    for t in range(T):
        for i in get_order(nsamples):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = perceptron_single_step_update(feature_vector, label, theta, theta_0)
            theta_sum += theta
            theta_0_sum += theta_0

    theta_avg = theta_sum / (T * nsamples)
    theta_0_avg = theta_0_sum / (T * nsamples)

    return theta_avg, theta_0_avg    

    raise NotImplementedError

"""
Explanation:

1. Initialization:
    - Initialize `theta` and `theta_0` to zero.
    - Initialize `theta_sum` and `theta_0_sum` to zero to keep track of the sum of `theta` and `theta_0` values over all iterations.

2. Training Loop:
    - Iterate through the dataset `T` times.
    - For each iteration, use the order specified by `get_order(nsamples)`.

3. Update Weights and Bias:
    - For each data point, update `theta` and `theta_0` using the `perceptron_single_step_update` function.
    - Add the updated `theta` and `theta_0` to `theta_sum` and `theta_0_sum`.

4. Compute Averages:
    - After completing all iterations, calculate the average `theta` and `theta_0` by dividing `theta_sum` and `theta_0_sum` by the total number of updates (`T * nsamples`).

5. Return Average Weights and Bias:
    - Return the averaged `theta` and `theta_0`.
"""


def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        theta,
        theta_0):
    """
    Updates the classification parameters `theta` and `theta_0` via a single
    step of the Pegasos algorithm.  Returns new parameters rather than
    modifying in-place.

    Args:
        `feature_vector` - A numpy array describing a single data point.
        `label` - The correct classification of the feature vector.
        `L` - The lamba value being used to update the parameters.
        `eta` - Learning rate to update parameters.
        `theta` - The old theta being used by the Pegasos
            algorithm before this update.
        `theta_0` - The old theta_0 being used by the
            Pegasos algorithm before this update.
    Returns:
        a tuple where the first element is a numpy array with the value of
        theta after the old update has completed and the second element is a
        real valued number with the value of theta_0 after the old updated has
        completed.
    """
    if label * (np.dot(theta, feature_vector) + theta_0) <= 1:
        # Update theta
        theta = (1 - eta * L) * theta + eta * label * feature_vector
        # Update theta_0
        theta_0 = theta_0 + eta * label
    else:
        # Only apply the regularization term
        theta = (1 - eta * L) * theta
    
    return theta, theta_0
    raise NotImplementedError



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T iterations
    through the data set, there is no need to worry about stopping early.  For
    each update, set learning rate = 1/sqrt(t), where t is a counter for the
    number of updates performed so far (between 1 and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.  Do
    not copy paste code from previous parts.

    Args:
        `feature_matrix` - A numpy matrix describing the given data. Each row
            represents a single data point.
        `labels` - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        `T` - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        `L` - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns:
        a tuple where the first element is a numpy array with the value of the
        theta, the linear classification parameter, found after T iterations
        through the feature matrix and the second element is a real number with
        the value of the theta_0, the offset classification parameter, found
        after T iterations through the feature matrix.
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0.0
    nsamples = feature_matrix.shape[0]
    t = 1

    for _ in range(T):
        for i in get_order(nsamples):
            eta = 1 / np.sqrt(t)
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta_0 = pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0)
            t += 1

    return theta, theta_0
    
    raise NotImplementedError



#==============================================================================
#===  PART II  ================================================================
#==============================================================================


"""
    #pragma: coderesponse template
def decision_function(feature_vector, theta, theta_0):
    return np.dot(theta, feature_vector) + theta_0

def classify_vector(feature_vector, theta, theta_0):
    return 2 * np.heaviside(decision_function(feature_vector, theta, theta_0), 0) - 1
    #pragma: coderesponse end
"""


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses given parameters to classify a set of
    data points.

    Args:
        `feature_matrix` - numpy matrix describing the given data. Each row
            represents a single data point.
        `theta` - numpy array describing the linear classifier.
        `theta_0` - real valued number representing the offset parameter.

    Returns:
        a numpy array of 1s and -1s where the kth element of the array is the
        predicted classification of the kth row of the feature matrix using the
        given theta and theta_0. If a prediction is GREATER THAN zero, it
        should be considered a positive classification.
    """
    def decision_function(feature_vector, theta, theta_0):
        return np.dot(theta, feature_vector) + theta_0

    def classify_vector(feature_vector, theta, theta_0):
        return 2 * np.heaviside(decision_function(feature_vector, theta, theta_0), 0) - 1
    
    return np.apply_along_axis(classify_vector, 1, feature_matrix, theta, theta_0)

    raise NotImplementedError

"""
Explanation:
Nested Functions:
- decision_function and classify_vector are defined within the classify function. 
  These functions are used to encapsulate the logic needed for the classification.

Classification Logic:
- decision_function computes the decision value for a single feature vector.
- classify_vector uses the decision_function to classify the feature vector as 1 or -1 based on whether the decision value is greater than zero.

Using np.apply_along_axis:
- The classify_vector function is applied to each row of the feature_matrix using np.apply_along_axis, 
  resulting in a numpy array of classifications.
"""



def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.  The classifier is
    trained on the train data.  The classifier's accuracy on the train and
    validation data is then returned.

    Args:
        `classifier` - A learning function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        `train_feature_matrix` - A numpy matrix describing the training
            data. Each row represents a single data point.
        `val_feature_matrix` - A numpy matrix describing the validation
            data. Each row represents a single data point.
        `train_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        `val_labels` - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        `kwargs` - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns:
        a tuple in which the first element is the (scalar) accuracy of the
        trained classifier on the training data and the second element is the
        accuracy of the trained classifier on the validation data.
    """
    # Your code here

    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    
    train_preds = classify(train_feature_matrix, theta, theta_0)
    val_preds = classify(val_feature_matrix, theta, theta_0)
    
    train_accuracy = accuracy(train_preds, train_labels)
    val_accuracy = accuracy(val_preds, val_labels)
    
    return train_accuracy, val_accuracy

    raise NotImplementedError

"""
Explanation:
1. Train the Classifier:
   - Call the classifier function with the train_feature_matrix, train_labels, and any additional kwargs.
   - This returns the trained parameters theta and theta_0.

2. Predict Labels:
   - Use the classify function to predict labels for both the training and validation data using the trained parameters.

3. Calculate Accuracy:
   - Use the accuracy function to calculate the accuracy of the predictions against the true labels for both the training and validation datasets.

4. Return Accuracies:
   - Return a tuple containing the training accuracy and the validation accuracy.

This function provides a framework to evaluate the performance of your classifiers on both training and validation data, helping to ensure that your model generalizes well to unseen data.
"""


def extract_words(text):
    """
    Helper function for `bag_of_words(...)`.
    Args:
        a string `text`.
    Returns:
        a list of lowercased words in the string, where punctuation and digits
        count as their own words.
    """
    # Your code here
    
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=True):
    """
    NOTE: feel free to change this code as guided by Section 3 (e.g. remove
    stopwords, add bigrams etc.)

    Args:
        `texts` - a list of natural language strings.
    Returns:
        a dictionary that maps each word appearing in `texts` to a unique
        integer `index`.
    """
    # Your code here

    stopword = set()
    if remove_stopword:
        with open('stopwords.txt', 'r') as f:
            stopword = set(f.read().split())

    indices_by_word = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_bow_feature_vectors(reviews, indices_by_word, binarize=False):
    """
    Args:
        `reviews` - a list of natural language strings
        `indices_by_word` - a dictionary of uniquely-indexed words.
    Returns:
        a matrix representing each review via bag-of-words features.  This
        matrix thus has shape (n, m), where n counts reviews and m counts words
        in the dictionary.
    """
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1

    if binarize:
        feature_matrix[feature_matrix > 0] = 1
    
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the fraction of predictions that are correct.
    """
    return (preds == targets).mean()
