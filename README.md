# netcom_dataScience_dataAnalytics
---

## Day 1
---
* Lab And Intro to Python
  * [data files](https://drive.google.com/file/d/1TtqZDek6TxC_hAHpMwK3TCJBVBlMZmTE/view?usp=sharing)
  * [vm](https://www.dropbox.com/s/sxh09e3ffnxkhon/data-science.ova?dl=0)
* Intermediate Python
* Intro to Machine Learning: Team Data Science Life Cycle
*  Question: Pertaining to [model leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)#cite_note-KaufmanKDD11-1): 
![towards data science example](https://www.google.com/url?sa=i&url=https%3A%2F%2Ftowardsdatascience.com%2Fhow-data-leakage-affects-machine-learning-models-in-practice-f448be6080d0&psig=AOvVaw0RtW9kwggaZmRfFlx-BGxA&ust=1623207674600000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCNimuvqFh_ECFQAAAAAdAAAAABAD)
> In statistics and machine learning, leakage (also known as data leakage or target leakage) is the use of information in the model training process which would not be expected to be available at prediction time, causing the predictive scores (metrics) to overestimate the model's utility when run in a production environment.
* R Code Example
```R
> x <- c(1, 5, 4, 9, 0)
> typeof(x)
[1] "double"
> length(x)
[1] 5
> x <- c(1, 5.4, TRUE, "hello")
> x
[1] "1"     "5.4"   "TRUE"  "hello"
> typeof(x)
[1] "character"
```
* Reversing a Python String
```python
# reversing a string, I forgot to use the `''.join.reverse()` or negative indexing with `[::-1]`
'a string'[::-1]
''.join(reversed('a string'))
```
* [Lambda Loop Pitfalls](https://nbviewer.jupyter.org/github/rasbt/python_reference/blob/master/tutorials/not_so_obvious_python_stuff.ipynb?create=1#lambda_closure)


## Day 2
----
**References for Questions Asked**:
  * [Pandas Apply Function](https://nbviewer.jupyter.org/github/rasbt/python_reference/blob/master/tutorials/things_in_pandas.ipynb#Applying-Computations-Rows-wise)
  * [PyCon on youtube: Hacking Nintendo Game](https://www.youtube.com/watch?v=v75rNdPukuI)
  * [Anatomy of Matplotlib Youtube SciPy 2018](https://www.youtube.com/watch?v=6gdNUDs6QPc&t=16s)
  * [Python's Infamous Gil](https://www.youtube.com/watch?v=KVKufdTphKs&t=1s)
  * [The Gilectomy](https://www.youtube.com/watch?v=P3AyI_u66Bw&t=74s)
  * [The Gilectomy: How It's Going](https://www.youtube.com/watch?v=pLqv11ScGsQ)
  * [Thinking Outside the GIL with AsincIO](https://www.youtube.com/watch?v=0kXaLh8Fz3k)

### **Pandas and Numpy Notebooks**
  * [Pandas Manual](https://pandas.pydata.org/pandas-docs/stable/pandas.pdf) 
  * [pandas - head() to tail()](https://www.youtube.com/watch?v=lkLl_QKLgcA)

### **Numpy**
 * [Numpy Array Ops Docs](https://numpy.org/doc/1.20/user/absolute_beginners.html)
 * [Numpy Tips and Tricks](https://nbviewer.jupyter.org/github/rasbt/python_reference/blob/master/tutorials/numpy_nan_quickguide.ipynb)
 * [3Brown1Blue Youtube](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw)
 * [Cleaning Data in Pandas Daniel Chen PyData 2018](https://www.youtube.com/watch?v=iYie42M1ZyU&t=1852s)
 * [Advanced Numpy](https://www.youtube.com/watch?v=poD8ud4MxOY&t=1262s)
 * [Pandas vs Koalas](https://www.youtube.com/watch?v=xcGEQUURAuk)

### **Seaborn notebook**
  * [State of Tools: Scipy 2015 Keynote](https://www.youtube.com/watch?v=5GlNDD7qbP4&t=1s)
    * [His Book](https://jakevdp.github.io/PythonDataScienceHandbook/)
  
## Day 3
-----
**References Related to Questions Asked**:
* [Virtual Env Management - Stack Overflow Dependency Hell](https://stackoverflow.com/questions/54475042/python-dependency-hell-a-compromise-between-virtualenv-and-global-dependencies)
* [Virtual Env Management - Reddit Answers](https://www.reddit.com/r/Python/comments/8clufh/how_do_you_manage_your_virtualenv/dxhdifx/)
  * A quote from my friend who is a python instructor as well:
  > I use venv for everything I'm going to show off, but my poor global state is masive and conflicts all the time
* [Crash Course in Applied Linear Algebra](https://www.youtube.com/watch?v=wkxgZirbCr4)
* Time Series Focused PostgreSQL [TimeScaleDB](https://www.timescale.com/)
* Blow the server up and still survive [CockroachDB](https://www.cockroachlabs.com/)
* Distributed Databases on a Raspberry Pi [Stalking a City for Frivolity](https://www.youtube.com/watch?v=ubjuWqUE9wQ)
* Combine **Data Lineage** with end 2 end Piplines [Pachyderm](https://www.pachyderm.com/)
* Container Metrics [Prometheus](https://prometheus.io/docs/guides/cadvisor/)
* Microservices Consideration [Orchestrating Chaos - QCon](https://www.youtube.com/watch?v=CZ3wIuvmHeM)
* AWS SAM Templates for Microserivices Boot Strap Testing [SAM tempate github](https://github.com/aws/aws-sam-cli-app-templates)
* CookieCutter Templating for Project Distributing [Cookie Cutter github](https://github.com/cookiecutter/cookiecutter)
* Consuming Models in PowerBi [Azure ML -> PowerBI](https://docs.microsoft.com/en-us/power-bi/connect-data/service-aml-integrate)
* [Create R based visuals in PowerBi](https://docs.microsoft.com/en-us/power-bi/create-reports/desktop-r-visuals)

### Intro to ML in Python - Categorical / Numerical
----
  * _How To Lie With Statistics Darry Huff_: `@google: how to lie with statistics`
  * [Winning With SImple, even Linear Models](https://www.youtube.com/watch?v=68ABAU_V8qI)
  * [Statistics Done Wrong - Alex Reinhart](https://www.statisticsdonewrong.com/)
  * [Stats for Hackers: Vanderplas Youtube](https://www.youtube.com/watch?v=Iq9DzN6mvYA&t=1992s)
  * [Statistical Thinking For Data Scientists](https://www.youtube.com/watch?v=TGGGDpb04Yc)
  * [All About That Bayes](https://www.youtube.com/watch?v=eDMGDhyDxuY)
  * [Everything Wrong with Statistics and How To Fix It](https://www.youtube.com/watch?v=be2wuOaglFY)

### SKLearn Con Notebooks:
----
[TODO] The `figure` modules is out of date and needs to be updated; 
[patch 1] `find ./ -type f -exec sed -i -e 's/from-missing-library/import fix/g' {} \;`

* [Recap of the SKLearn API and which Estimator has which output](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/09.Review_of_Scikit-learn_API.ipynb)
  * Some models do have the `model.transform` method attached to them once they are fit; `sklearn.preprocessing` has this enabling you to create processing pipelines and clean data at scale; Other models will also have `.transform` showing more findings by the model that may help for visualization, etc.
  * Other models will give you _denisty estimation_, which is a measure of how close the data follows the _structure_ models. i.e. I see a new computer, it's missing installed security services, out of date apps -> **evil** 
* ~stratified split explainer~
* [Supervised Learning 1 - Classification](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/05.Supervised_Learning-Classification.ipynb)
 * Includes Logistic Regressions, a breakdown of which can be seen in [3- Logistic Regression and Naivve Bayes](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/3-Logistic%20Regression%20and%20Naive%20Bayes.ipynb)
 * There is a bug in the data for these notesbooks. I'm either going to write a DF generator to handle making the data sets with NP so the missing data is no longer an issue, or write every example using `sklearn.datasets.make_*` to generate the sets with matches columns. 
* [Where does the weight value come from in OLS formula](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/06.Supervised_Learning-Regression.ipynb)
  * The weight we are referring to here is the coefficient to multiply the data by
* [Unsupervised Leaning part one: PCA Dimension Reduction](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/07.Unsupervised_Learning-Transformations_and_Dimensionality_Reduction.ipynb)
  *  How do I find the `n_components` for my data? The value can take many python data types, but passing in no arguments uses the baked in logic from sklearn to figure out what's optimal. Changing the values used in `PCA()` and checking the `pca.explained_variance_ratio` will let you gather some insight into what the PCA model believes in captured by Component 1 and Component 2.
* **[Cluster: Why is the computed accuray 0.0 and not 1.0, how to fix it](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/08.Unsupervised_Learning-Clustering.ipynb)
  * What this questions is trying to get across is that, the accuracy isn't the score to use for cluserting, none of the labels are corresponding in the example. Checking the confusion matrix shows all the labels of the classes are classified by the slice `[class0,class1,class2]` which matches the data, however, cluster membership doesn't match the labels. _Colorblind_ to the label, but can see the _hue_.
  * How to fix this? Numerous ways can be checked for distances measures between clusters, rotating clusters can help, but some other model would have to come before this one. Instead of focusing on the labels, focusing on the data points and pairs of data points that are preseved post processing. The `adjusted_rand_score` in the example follows this similar methodology, and the score goes to `1.0`


  * [Validation and Model Selection - pyData2015](https://nbviewer.jupyter.org/github/jakevdp/sklearn_pydata2015/blob/master/notebooks/05-Validation.ipynb)
  * [Advanced SciKit Learn](https://www.youtube.com/watch?v=ZL77pbWBZQA&t=1078s) 
  * [SkLearn ROC Curve Visualiztion API](https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_roc_curve_visualization_api.html#sphx-glr-auto-examples-miscellaneous-plot-roc-curve-visualization-api-py)
  * [ROC Curves Explained](https://mlwhiz.com/blog/2021/02/03/roc-auc-curves-explained/)
  * [How to evaluate K-Modes Cluster](https://datascience.stackexchange.com/questions/64455/how-to-evaluate-the-k-modes-clusters)
  * [What does the '5' mean in the `CountVectorizer.vocabulary`](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/11.Text_Feature_Extraction.ipynb)
    * This is a dictionary structure as one of the students was pointing out trying to help me, the 5 is a positional argument to access the word in the dictionary. The `.vocabulary_` value represents the unique list of words found that is assigned a unique token ( 5 here ); Valuable information here is `len(.vocabulary_)`. It should be the link between the data strucutre and the tokenized form. Each word is assigned an arbitrary dimension ( 5 )
    * In our example, I was wrong, and this is the assigned position, not the count.  
    * tokenized -> the white space and punctuation has been stripped ).
    * Why use a sparse matrix? A lot of the vocabulary results will have 0's as results given a large dictionary
* [Data Processing and Regressions - Titanic Case Study](https://github.com/t-0-m-1-3/netcom_dataScience_dataAnalytics/blob/main/Day3/scipy-2018-sklearn-master/notebooks/10.Case_Study-Titanic_Survival.ipynb)
  * Encoding Data Categorically with `pd.get_dummies(data,columns=['list','to','mask']`
  * `sklearn.impute` is used here to scale the data for passing into the Random Forest
  * `DummyClassifier` will count the number of times it sees [0,1] and predicts the majority class, just looking at `Y`; **baccarat** or **Constant Classifier**.   
  * **Development Branch of SKLearn** has a Column Transformer that handles multiple data type datasets, cutting out some of `pandas` lift here. 

* Logistic Regression and Naieve Bayes
  * Here is where I ran once, the Text example failed, showing off a Bayesian Type model, and the `20_newsgrounps` dataset, to predict the labels based on new text. 

* KNN

* Cross Validation -> Splitting in multiple ways, with different % of the data to see how that affects training and test results. 
* 

### Day 4
----
* Question Recap
* 14-model-complexity and grid search 
* 15-Pipelining Estimators
* 16-Performance Metrics and Model Eval
~* 17-In Depth Linear~
~* 18-In Depth Tree and Forests~
~* 19-Feature Selection~
~* 20-Heriarchical and Density Clustering~

~* Ensemble Models~
* NLTK ( 2-1 -> maybe 5-1 )
* Association Rules


### Day 5
----
* ANN / Perceptron Build
* Hadoop / Spark-PySpark
* Demo 

* References and Resources Mentioned:
  * [python main website](https://www.python.org)
  * [Keynote: Guido Van Rossum](https://www.youtube.com/watch?v=wf-BqAjZb8M)
  * [Keynote: Perry Greenfield How Python Found its way into Astronomy](https://www.youtube.com/watch?v=uz53IV1V_Xo&t=1630s)
  * [violent python in python 3](https://github.com/EONRaider/violent-python3)
    * **Intersting Google Dork** `(@google:(github:violent python) & (filetype:pdf))`
  * [ReGex By Al Sweigert](https://www.youtube.com/watch?v=abrcJ9MpF60)
    * [His Book](https://automatetheboringstuff.com/2e/chapter0/) 
  * [Engineer Man on Youtube's Python Series](https://www.youtube.com/watch?v=VQxBd5tLza8&list=PLlcnQQJK8SUjW_HiBWhZ_XOfCq9Hu0aeY)
  * [Uncle Bob Martin: The Future Of Programm Youtube](https://www.youtube.com/watch?v=ecIWPzGEbFc)
  * [Probablistic Programming and Bayesian Modeling with PyMC3](https://www.youtube.com/watch?v=M-kBB2I4QlE&t=1316s)
  * [Ten Ways To Fizz Buzz Joel Grus](https://www.youtube.com/watch?v=E7JAIF9FOnM)
  * [RaspberryPi Python Games](https://www.raspberrypi.org/documentation/usage/python-games/)
  
