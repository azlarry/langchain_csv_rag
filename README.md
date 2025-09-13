# Langchain with RAG using CSV data

This is an exploratory and demonstration project that uses Langchain and Retrieval-Augmented Generation (RAG) to leverage data in a CSV file to improve Large Language Model (LLM) responses.

## Built With

* Ollama locally-hosted gpt-oss:20b LLM
* Langchain with [Pandas dataframe agent](https://python.langchain.com/api_reference/experimental/agents/langchain_experimental.agents.agent_toolkits.csv.base.create_csv_agent.html)
* NFL player data from https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.csv

## Overview

The goal of this project is to ask the LLM two questions related to NFL player performance (discussed below) and see if the responses improve if supporting data is provided to the LLM via RAG.  For each question, the following process was used:

* The correct answer is determined using manual methods
* The question is provided to the LLM with no supporting data (intended to validate that the LLM does not have the data and cannot answer the question properly)
* The question is provided to the same LLM but with RAG and a CSV file that contains the data needed to answer the question

Overall the LLM without RAG data was unable to answer the questions, but was able to with RAG data.

## Discussion

### CSV Data

The CSV data used is available [here](https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.csv).  This data is for week 1 of the 2025 NFL season.  Each row represents a single player with statistics about their performance in that week's game.

The relevant columns of data for the project questions are as follows:

* PlayerName
* Team
* ReceivingTD

It is important to note that the LLM needs to interpret several aspects from the question, e.g. relating a player to the PlayerName column and relating the number of receiving touchdowns they made to the ReceivingTD column.  There is no contextual, prefix, or other data provided to indicate to the LLM what each column means.  It is also important to note that there are several similar columns that could be misinterpreted, e.g. PassingTD, ReceivingRec, ReceivingYDS, RetTD, FumTD, TouchReceptions, and others.

### Question One

The first question is:

***Which player had the most receiving touchdowns in 2025?***

The answer can be manually determined as follows:

```
player_receiving_tds = df.sort_values(by='ReceivingTD', ascending=False)

print(f"ANSWER: Player with most receiving TDs: {player_receiving_tds.head(1)['PlayerName'].values[0]} with {player_receiving_tds.head(1)['ReceivingTD'].values[0]} TDs")
```

with the correct answer output being:

```
ANSWER: Player with most receiving TDs: Emeka Egbuka with 2.0 TDs
```

Providing the question to the LLM without the RAG data, the response was as follows:

```
I’m sorry, but I don’t have that information.
```

This validates that the LLM was not trained on, and otherwise does not have access to the recent NFL player data.

Providing the same question to the same LLM with the RAG CSV data, the response was:

```
The highest number of receiving touchdowns recorded in the 2025 season is **2**.

Both **Quentin Johnston** (LAC) and **Emeka Egbuka** (TB) achieved that total, so they are tied for the most receiving touchdowns that year.
```

This is the correct answer, and is actually better than the manually determined answer which missed the second player with the same number of receiving touchdowns.

This question is fairly straightforward for the LLM with RAG CSV data to answer, as it requires simply looking up the data and determining the max value.

### Question Two

The second question is:

***Which team had the most receiving touchdowns in 2025?***

The answer can be manually determined as follows:

```
team_receiving_tds = df.groupby('Team')['ReceivingTD'].sum().sort_values(ascending=False)

print(f"ANSWER: Team with most receiving TDs: {team_receiving_tds.index[0]} with {team_receiving_tds.iloc[0]} TDs")
```

with the correct answer output being:

```
ANSWER: Team with most receiving TDs: LAC with 3.0 TDs
```

Providing the question to the LLM without the RAG data, the response was as follows:

```
I’m sorry, but I don’t have information on the 2025 NFL season.
```

As with the first question, this response validates that the LLM was not trained on, and otherwise does not have access to the recent NFL player data.

Providing the same question to the same LLM with the RAG CSV data, the response was:

```
The team with the highest total of receiving touchdowns in the 2025 season was the **Los Angeles Chargers (LAC)**, with **3 receiving touchdowns**
```

This is the correct answer.

This question requires more analysis for the LLM with RAG CSV data to answer, in that it had to do several things to arrive at the correct answer:

* group all of the players by team
* determine the total receiving touchdowns for each team
* find the max value of this total

## Early iterations

This project was initially attempted with the llama3.2 1B model, but it was unable to correctly answer the questions.  For the first question, it seems it was unable to properly associate the CSV columns with the desired data.  When a prompt was used to 'help' the LLM understand the CSV data it was still unable to correctly answer the second question.  Whether this is primarily due to the smaller size of the model (2GB vs 13GB for gpt-oss:20b) is not known.

## Conclusion and future work

This was an interesting project that I think clearly demonstrates the capability of RAG to add new data to an LLM.

* exploring an entire season of data and conduct more in-depth analysis
* explore limitations of prohibiting the agent from exeucting Python code i.e. the ```allow_dangerous_code = False``` agent parameter
* similarly, explore precautions/safeguards if necessary to have ```allow_dangerous_code = True```
* evaluate the performance and capability of different models and model sizes
* explore variations of providing prompt/context explanations of the CSV data



