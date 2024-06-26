# Sentiment Analysis Project

This repository contains the code and resources for the Sentiment Analysis Project, which involves building a classifier that labels reviews as positive or negative using text-based features and various linear classifiers.

## Project Structure

The project is structured as follows:

- `utils.py`: Contains utility functions for data loading, writing predictions, plotting data, and tuning classifiers.
- `test.py`: Contains tests to verify the correctness of various functions implemented in the project.
- `project1.py`: Contains the main implementation of the linear classifiers and their respective algorithms.
- `main.py`: The main script to run the training, validation, and testing of the classifiers.

## Data

The dataset consists of several reviews, each labeled with -1 (negative) or +1 (positive). The data is split into training, validation, and test sets:

- `reviews_train.tsv`
- `reviews_validation.tsv`
- `reviews_test.tsv`

## Classifiers

The project implements the following classifiers:

1. **Perceptron**
2. **Average Perceptron**
3. **Pegasos (SVM)**

## Usage

To run the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```

2. **Install dependencies:**

   Make sure you have Python and the required libraries installed. You can use the following command to install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main script:**

   ```bash
   python main.py
   ```

## Hyperparameter Tuning

The hyperparameters (T and λ) are tuned using grid search. The optimal values are selected based on validation accuracy.

## Feature Engineering

- **Bag of Words:** The reviews are converted into feature vectors using a Bag of Words approach.
- **Stopwords Removal:** Stopwords are removed to improve the performance of the classifier.
- **Count Features:** The feature vectors are modified to use the count of each word in the document rather than binary indicators.

## Results

The results of the classifiers are printed in the console. The most explanatory word features are also displayed.

### Example Output

```bash
Training accuracy for perceptron:   0.8157
Validation accuracy for perceptron: 0.7160
Training accuracy for average perceptron:   0.9728
Validation accuracy for average perceptron: 0.7980
Training accuracy for Pegasos:                     0.9143
Validation accuracy for Pegasos:                   0.7900

Test accuracy for Pegasos: 0.8020
Most Explanatory Word Features:
['delicious', 'great', '!', 'best', 'perfect', 'loves', 'wonderful', 'glad', 'love', 'quickly']
```

## Acknowledgments

I would like to thank the MIT faculty for their excellent instruction and guidance throughout the [Statistics and Data Science MicroMasters Program](https://micromasters.mit.edu/ds/), particularly in the course "Machine Learning with Python: from Linear Models to Deep Learning".

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
