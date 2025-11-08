# Investigating In-Context Learning of Linear Regression in Transformers

This repository attempts to reproduce and extend the results from _What Can Transformers Learn In Context? A Case Study of Simple Functions?_ (Garg et al. 2023) and _Transformers Learn In-Context by Gradient Descent_ (Oswald et al. 2023). Specifically, I aim to show that transformers are able to achieve comparable performance to least squares for standard linear regression, LASSO for sparse linear regression, and gradient descent for overparameterized linear regression. 

The repository is structured as follows:

`experiments-from-scratch`, this folder contains all of the code that I've written   
---| `overparameterized-gradient-descent`, this folder contains the code to show that gradient descent is able to achieve zero training loss on an overparameterized linear regression dataset  
---| `gradient-descent-lsa`, this folder contains the code to generate synthetic training data and train a transformer using curriculum learning  
`in-context-learning`, this is the codebase provided by the authors of _What Can Transformers Learn In Context? A Case Study of Simple Functions?_  
`transformers-learn-in-context-by-gradient-descent`, this is the codebase provided by the authors of _Transformers Learn In-Context by Gradient Descent_. 