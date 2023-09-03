""" This is the first agent llm with the duckduckgo search wrapper

This uses the search engine duck duck go, if the AI thinks its a good
thing todo. The setup is very straight forward.

The result is `verbose`, which prints out the actions the agent is doing
when solving the input.

Results
```

> Entering new AgentExecutor chain...

I should search for the latest news on zeit.de.
Action Input: query=latestNews
Observation: Invalid Format: Missing 'Action:' after 'Thought:
Thought:

I will search for the latest news on zeit.de using the DuckDuckGo Search wrapper.
Action Input: https://duckduckgo.com/u?q=zeit.de%20latestNews&ia=news
Observation: Invalid Format: Missing 'Action:' after 'Thought:
Thought:

I will search for the latest news on zeit.de using the DuckDuckGo Search wrapper.
Action Input: https://duckduckgo.com/u?q=zeit.de%20latestNews&ia=news
Final Answer: Latest news headline on zeit.de is "German government launches €75 billion stimulus package to support economy".

> Finished chain.
Latest news headline on zeit.de is "German government launches €75 billion stimulus package to support economy".
´´´

It performs the task 3 times, as its the default retry.

"""

# llamacpp
from langchain.llms import LlamaCpp
# agent and tools import
from langchain.agents import initialize_agent, AgentType
from langchain.tools import DuckDuckGoSearchRun

# PATH where the model is on your device
MODEL_PATH = "/home/andreas/development/ai/models/llama-2-7b-chat.ggmlv3.q4_0.bin"
MODEL_PARAMS = {
    "model_path": MODEL_PATH,
    "n_ctx": 2048, # context size
    "n_gpu_layers": 10, # layers to shift to qpu if possible
}

# init the LLM
llm = LlamaCpp(**MODEL_PARAMS)

search = DuckDuckGoSearchRun(name="Search")
search_agent = initialize_agent(
    [search],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=True, # verbose let the agent print out what it is doing
)
response = search_agent.run("What is the headline on zeit.de?")
print(response)
