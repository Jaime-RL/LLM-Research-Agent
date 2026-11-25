# Research Agent (LangChain + Llama 3.1 8B)

A command-line conversational agent built with LangChain. It uses `meta-llama/Llama-3.1-8B-Instruct` through Hugging Face Inference (`HuggingFaceEndpoint` + `ChatHuggingFace`) and gives the model one tool: a Wikipedia search (`WikipediaQueryRun`).

The system prompt (in Spanish) tells the agent to answer clearly, to look up people, places, concepts and events on Wikipedia, not to make things up, and to say so when the tool does not return enough information.

## How it works

1. Checks that `HUGGINGFACEHUB_API_TOKEN` is set, and exits with a message if it is not.
2. Creates the model endpoint (`temperature=0.1`, `max_new_tokens=512`, `provider="auto"`).
3. Sets up the Wikipedia tool (top 1 result, up to 3,000 characters).
4. Builds the agent with `langchain.agents.create_agent`.
5. Runs a loop: reads a question, calls the agent and prints the final answer. Type `salir`, `exit` or `quit` to stop.

## Requirements

- Python and the pinned dependencies in `requirements.txt`
- A Hugging Face access token with Inference access to `meta-llama/Llama-3.1-8B-Instruct` (a gated model: you must accept its licence on Hugging Face)

## Usage

```bash
pip install -r requirements.txt
export HUGGINGFACEHUB_API_TOKEN='your_token'
python agente_investigador.py
```

Example:

```
Pregunta para el agente: ¿Quién fue Ada Lovelace?
```

The agent decides whether to query Wikipedia and prints its final answer between separator lines.

## Limitations

- One tool only (Wikipedia), with a single result truncated to 3,000 characters.
- No memory between questions: each question is sent to the agent on its own.
- Answers are in Spanish (set in the system prompt).
- Depends on Hugging Face Inference being available for the model and on your token's quota. Tool calling depends on the provider chosen by `provider="auto"`.
