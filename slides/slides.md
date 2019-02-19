---
theme : "black"
transition: "zoom"
highlightTheme: "darkula"
separator: ^---$
verticalSeparator: ^--$
---

#### .NET BA 02/2019

# Introduction to ML.NET

#### by Marek Linka

--

## About me

* Senior software engineer @ ERNI
* Focus on up-and-coming .NET (Core, ASP.NET, ML.NET)
* Machine learning enthusiast and driver
* CoreFX and ML.NET contributor
* Debugger of the strange :)

----

https://twitter.com/mareklinka

---

## What is ML.NET

--

### ML.NET is...

* A machine learning framework for creating and using machine learning models in .NET
* Cross-platform
    * Windows
    * Linux
    * macOS
* Open-source on GitHub

--

### A little history

* Originally developed by Microsoft Research for internal use
* Later open-sourced on GitHub as a library
    * .NET Core and .NET Standard
    * Currently 10 public preview releases
    * Version 0.1 released in May 2018
* Under active development
* Community contributions very welcome

--

### Features

* Data loading and transformation
* Supervised learning
* Unsupervised learning
* Model evaluation
* Model explanation
* Scoring of pre-trained models (Onnx, TensorFlow)

--

### Supported use-cases

* Standard regression/classification
* Text processing
* Image processing
* Anomaly detection
* Ranking
* Recommendation

--

### What's not there (yet)

* Deep learning (training)
* Distributed training
* ARM/x86 support
* __Stable API__
* __Stable and thorough documentation__

---

## Using ML.NET

--

### Training

* Decide what you want to do
* Gather data
* Define your ML pipeline
    * Step 1: Load data
    * Step 2: Transform data (group, encode, fill missing values)
    * Step 3: Add a learning algorithm
    * Step 4: Train
    * Step 5: Evaluate the model
    * Step 6: Save the model for later scoring

--

### Scoring

* Step 1: Add your trained model to your application
* Step 2: Load your model
* Step 3: Feed data into your model and get predictions
* Step 4: PROFIT

---

#### DEMO
## Learning from disaster
#### (Binary classification)

--

### What we saw

* How to add ML.NET to your app
* Loading, cleaning, and transforming data
* Creating a learning pipeline
* Training and evaluating a simple model

---

#### DEMO
## Classifying disease
#### (Using pre-trained TensorFlow model in c#)

--

## What we saw

* Adding TensorFlow scoring to your app
* Loading a Keras-trained TF model
* Scoring a deep learning model directly from C# code

---

### A call to action

* ML.NET has fantastic potential for us, .NET programmers
* ML is all the rage, now we can incorporate it in our apps with very little friction
* It's open-source - go contribute!
    * The repo has _up-for-grabs_ and _good-first-issue_ tags
    * The maintainers are friendly and open
    * Every little bit helps - propose features, add documentation, implement stuff

--

## Q&A
#### (Ask me 3 or more questions)

--

### Resources

* https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet
* https://github.com/dotnet/machinelearning
* https://www.kaggle.com/c/titanic
* https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria/kernels
* https://www.coursera.org/specializations/deep-learning

--

# Thank you