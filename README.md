# ml-python
This repo contains Python examples for the Machine Learning Course at the [MSc Program of AI](http://msc-ai.iit.demokritos.gr), organzied by NCSR Demokritos and University of Piraeus. Most examples are in the form of Jupyter notebooks. Since github does not render plotly figures, the links of this readme file are not directly to the github files but to nbviewer links ([https://nbviewer.jupyter.org](https://nbviewer.jupyter.org)). 

## Topics

### Linear Regression
[notebooks/1-linear-regression.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/1-linear-regression.ipynb) demonstrates how Gradient Descent can be used to calculate a linear regression model, for a small dataset. Also, we plot the cost function (as a 2-D function of w and b) and the path followed by the Gradient Descent iteration. 

### Logistic Regression
[notebooks/2-logistic-regression.ipyn](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/2-logistic-regression.ipynb) demonstrates how to use Gradient Ascent to train a logistic regression binary classifier using some 2D training data. It also visualizes the decision surface and the probability estimates of the classifier on the 2D feature space. 

### Bayesian / GDA classifiers, Naive Bayes, kNN
[notebooks/3-gda.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/3-gda.ipynb) shows a simple example of how Gaussian Discriminant Analysis and the Bayes Rule can be used to estimate p(y|x) given y(x|y) for a 1-D binary classification task. 

### Perceptron, SVMs
[scripts/perceptron_demo.py](https://github.com/tyiannak/ml-python/blob/main/scripts/perceptron_demo.py) demonstrates the steps of the basic perceptron algorithm in an interactive manner using a dash-generated UI. 

### Decision Trees, Ensembles
[notebooks/8-decision-tree-example.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/8-decision-tree-example.ipynb) shows a Decision Tree training example, visualizing the decision tree areas and the tree itself. 

### Practical ML Issues
[notebooks/4-a_simple_sk_learn_example_songs_.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/4-a_simple_sk_learn_example_songs_.ipynb) demonstrates how to predict a song's popularity from the respective emotional valence and arousal using 2 features and a linear SVM classifier. Towards this end, a sample dataset of 5000 songs and respective Spotify metadata is used. 

[notebooks/4-b_simple_sk_learn_example_songs_.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/4-b_simple_sk_learn_example_songs_.ipynb) uses the same spotify dataset to predict a musical gender from the respective "song speechiness" (a spotify attribute) and emotional arousal (energy).

[notebooks/5-overfitting.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/5-overfitting.ipynb) and [notebooks/6-overfitting_class.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/6-overfitting_class.ipynb) demonstrate the problem of overfitting in regression and classfication respectively. 

[notebooks/7-ml_speed.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/7-ml_speed.ipynb) evaluate basic classifiers (including SVMs, Adaboost, Random Forestes, Naive Bayes, etc) with regards to their performance and speed (during training and testing), for various datasets with different number of dimensions (features), feature dependence and number of examples. Useful conclusions can be directly drawn by the results, e.g. Naive Bayes is always the fastest (both in training and testing time) and it has a good performance when features are indepedent. 

[notebooks/7-b-scaling.ipynb](https://nbviewer.org/github/tyiannak/ml-python/blob/main/notebooks/7-b-scaling.ipynb) shows how to use sklearn's standard scaler, to scale the features using their means and stds. The "wine" sklearn sample dataset is used and performance results with and without scaling are presented, for basic classifiers. 

[notebooks/9-evaluation.ipynb](https://nbviewer.jupyter.org/github/tyiannak/ml-python/blob/main/notebooks/9-evaluation.ipynb) shows how to compute classifier validation metrics (confusion matrix, accuracy, f1 etc) using scikit-learn. It also demonstrates the calculation of ROC and Precision-Recall curves on the 5000 song dataset described above. 

[notebooks/10-evaluation2.ipynb](https://nbviewer.org/github/tyiannak/ml-python/blob/main/notebooks/10-evaluation2.ipynb) demonstrates how to evaluate a classifier using either sklearn pipeline or "manually" using `sklearn.model_selection.RepeatedKFold`

### Clustering, Dimensionality Reduction
[scripts/kmeans_demo.py](https://github.com/tyiannak/ml-python/blob/main/scripts/kmeans_demo.py) uses dash to create a simple demo that shows how the kmeans algorithm works on artificial datasets

### Future tasks
#### Naive bayes for text task
#### Feature selection example
#### SVM example

## Author
<img src="https://tyiannak.github.io/files/3.JPG" align="left" height="100"/>

[Theodoros Giannakopoulos](https://tyiannak.github.io),
Principal Researcher of Multimodal Machine Learning at the [Multimedia Analysis Group of the Computational Intelligence Lab (MagCIL)](https://labs-repos.iit.demokritos.gr/MagCIL/index.html) of the Institute of Informatics and Telecommunications, of the National Center for Scientific Research "Demokritos"
