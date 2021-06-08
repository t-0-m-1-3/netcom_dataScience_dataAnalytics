# netcom_dataScience_dataAnalytics
---
Day 1
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

Day 2
* pandas - head() to tail()
* Numpy Chapter of jvd Book
* Seaborn notebook 

Day 3
* Intro to ML in Python - Categorical / Numerical
* Data Processing and Regressions
* Logistic Regression and Naieve Bayes
* KNN 

Day 4
* Decision Tree
* Ensemble Models
* Feature Extraction 

Day 5
* ANN / Perceptron Build
* Hadoop / Spark-PySpark
* Demo 

* References and Resources Mentioned:
  * [python main website](https://www.python.org)
  * [Keynote: Guido Van Rossum](https://www.youtube.com/watch?v=wf-BqAjZb8M)
  * [Keynote: Perry Greenfield How Python Found its way into Astronomy](https://www.youtube.com/watch?v=uz53IV1V_Xo&t=1630s)
  * [State of Tools: Scipy 2015 Keynote](https://www.youtube.com/watch?v=5GlNDD7qbP4&t=1s)
    * [His Book](https://jakevdp.github.io/PythonDataScienceHandbook/)
  * [violent python in python 3](https://github.com/EONRaider/violent-python3)
    * [Intersting Google Dork](@google:(github:violent python) & (filetype:pdf))
  * [ReGex By Al Sweigert](https://www.youtube.com/watch?v=abrcJ9MpF60)
    * [His Book](https://automatetheboringstuff.com/2e/chapter0/) 
  * [PyCon on youtube: Hacking Nintendo Game](https://www.youtube.com/watch?v=v75rNdPukuI)
  * [Stats for Hackers: Vanderplas Youtube](https://www.youtube.com/watch?v=Iq9DzN6mvYA&t=1992s)
  * [Statistical Thinking For Data Scientists](https://www.youtube.com/watch?v=TGGGDpb04Yc)
  * [All About That Bayes](https://www.youtube.com/watch?v=eDMGDhyDxuY)
  * [Everything Wrong with Statistics and How To Fix It](https://www.youtube.com/watch?v=be2wuOaglFY)
  * [Anatomy of Matplotlib Youtube SciPy 2018](https://www.youtube.com/watch?v=6gdNUDs6QPc&t=16s)
  * [Cleaning Data in Pandas Daniel Chen PyData 2018](https://www.youtube.com/watch?v=iYie42M1ZyU&t=1852s)
  * [Advanced Numpy](https://www.youtube.com/watch?v=poD8ud4MxOY&t=1262s)
  * [Python's Infamous Gil](https://www.youtube.com/watch?v=KVKufdTphKs&t=1s)
  * [The Gilectomy](https://www.youtube.com/watch?v=P3AyI_u66Bw&t=74s)
  * [The Gilectomy: How It's Going](https://www.youtube.com/watch?v=pLqv11ScGsQ)
  * [Thinking Outside the GIL with AsincIO](https://www.youtube.com/watch?v=0kXaLh8Fz3k)
  * [Engineer Man on Youtube's Python Series](https://www.youtube.com/watch?v=VQxBd5tLza8&list=PLlcnQQJK8SUjW_HiBWhZ_XOfCq9Hu0aeY)
  * [Uncle Bob Martin: The Future Of Programm Youtube](https://www.youtube.com/watch?v=ecIWPzGEbFc)
  * [Probablistic Programming and Bayesian Modeling with PyMC3](https://www.youtube.com/watch?v=M-kBB2I4QlE&t=1316s)
  * [Ten Ways To Fizz Buzz Joel Grus](https://www.youtube.com/watch?v=E7JAIF9FOnM)
  * [RaspberryPi Python Games](https://www.raspberrypi.org/documentation/usage/python-games/)
  
