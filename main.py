from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd


def get_non_rag_response(question, llm):

    response = llm.invoke(question)
    print(f"\n{question}\n")
    print(f"\nLLM:\n{response}\n")
    print("\n--------------------------------\n")
    return response


def get_rag_response(question, llm, df):

    agent = create_pandas_dataframe_agent(
        llm,
        df,
        prefix="You are a sports data analyst.  Use the provided data containing NFL player statistics for the 2025 season to answer the question as accurately as possible. If the data does not contain the information needed to answer the question, respond with 'I don't know based on the provided data.'",
        agent_type="tool-calling",
        allow_dangerous_code=True,
        verbose=True
    )

    response = agent.invoke(question)
    print(f"\n{question}\n")
    print(f"\nLLM with RAG:\n{response}\n")
    print("\n--------------------------------\n")
    return response


def main():
   
    llm = ChatOllama(
        model="gpt-oss:20b",
        temperature=0 # lower/0 temperature provides more deterministic or less 'creative' responses
    )

    question = "Which player had the most receiving touchdowns in 2025?"
    print(question)

    # check the answer without LLM/RAG
    df = pd.read_csv('WR.csv') # data from https://github.com/hvpkod/NFL-Data/blob/main/NFL-data-Players/2025/1/WR.csv
    player_receiving_tds = df.sort_values(by='ReceivingTD', ascending=False)
    print(f"ANSWER: Player with most receiving TDs: {player_receiving_tds.head(1)['PlayerName'].values[0]} with {player_receiving_tds.head(1)['ReceivingTD'].values[0]} TDs")
    
    response = get_non_rag_response(question, llm)
    response = get_rag_response(question, llm, df)

    question = "Which team had the most receiving touchdowns in 2025?"
    print(question)

    # check the answer without LLM/RAG
    team_receiving_tds = df.groupby('Team')['ReceivingTD'].sum().sort_values(ascending=False)
    print(team_receiving_tds.index[0])
    print(team_receiving_tds.iloc[0])
    print(f"ANSWER: Team with most receiving TDs: {team_receiving_tds.index[0]} with {team_receiving_tds.iloc[0]} TDs")

    response = get_non_rag_response(question, llm)
    response = get_rag_response(question, llm, df)


if __name__ == "__main__":
    main()