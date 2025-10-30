# Code to assist with NLP 2025 assignment

In this repository you will find the necessary resources to complete the NLP
assignment.


## Python environment

* The details of the **python** environment is provided in ```pyproject.toml```.

> [!WARNING]
> The code will be run in an environment based on the ```pyproject.toml``` file.
> Use of any extra libraries beyond those used in this repository might be
> **disqualifying** (i.e., will be marked 0 if it does not run). No extra
> libraries will be installed for evaluation purposes. This repository has been
> tested to work with ```python==3.13```.

## Writing an API call

* You would first need to create an account at [OpenRouter](https://openrouter.ai/) to get access to (free) LLMs via an API. An API key needs to be created.
* Choose a (free) model from the list of available models.

> [!NOTE]
> These two strings: your **API key** and the **name of the model** is needed
for the next step.

* A basic skeleton code to get a response to queries from an LLM is provided in ```api_call.py```


## Obtaining search results

* Code for this has been provided in ```search.py```. The code does the following:
  - Gets search results from **DuckDuckGo** corresponding to a query.
  - Selects the first **wikipedia** link.
  - Parses the **wikipedia** page and obtains the body text.

