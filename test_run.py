import json
from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

# next we can define our llm model to use
import os
from api_keys import GROQ_API_KEY, TAVILY_KEY
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['TAVILY_API_KEY'] = TAVILY_KEY

llm_model = init_chat_model(model="llama3-8b-8192", model_provider = "groq")

tavily_search_tool = TavilySearch(
    max_results=2,
    topic="general",
)

llm_tools = [tavily_search_tool]

# binding this tool to llm 
llm_with_search = llm_model.bind_tools(llm_tools)

# since state is the most important piece in the flow, we define the state class first

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # state holds the list of messages, when new message is given to state, 
    # reducer function (add_messages), append that message to the list
    # keeping all the previous conversation 
    # if no reducer fn, then it overwrites, instead of appending

def chat_node(state: State):
    # this node will take the stored messages in state and run 
    # chat llm on it
    print("into chat node")
    result = {'messages': [llm_with_search.invoke(state["messages"])]}
    print("chat node result: ", result)
    return result

class ToolNode:

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        print("into calling tool node")
        messages = inputs.get("messages", [])
        if messages:
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
            
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        print("tool node out generated: ", outputs)
        return {"messages": outputs}

def node_routing(state: State):
    message = state.get("messages", [])[-1]

    # if last message has tool calls from llm
    if hasattr(message, "tool_calls") and len(message.tool_calls) >0:
        print("routing to tool node")
        return "tools"
    return "end"

def run_graph(user_input: str):
    for event in graph.stream(
        {"messages": [HumanMessage(content = user_input)]}):
        for value in event.values():
            # printing latest message in state
            print("Chat: {}".format(value["messages"][-1].content))


# building graph
graph_builder = StateGraph(State)

# the first node would be the start node which will then point to chat node
graph_builder.add_edge(START, "chat_node")

# add this node to the graph, first argument would be name of node
graph_builder.add_node("chat_node", chat_node)

# here the last dictionary is for reference of which node to call, depending on 
# response from routing function
graph_builder.add_conditional_edges("chat_node",
                                    node_routing,
                                    {"tools" : "tool_node", "end": END})
# adding this tool node to graph
tool_node = ToolNode(tools = llm_tools)
graph_builder.add_node("tool_node", tool_node)


# in case the conditional edge chooses tool calling node, 
# we would want to return back to the llm 
graph_builder.add_edge("tool_node", "chat_node")

graph = graph_builder.compile()

# running the chat
while True:
    user_input = str(input("User: "))
    if user_input.lower() in ["quit", "q", "exit"]:
        break
    run_graph(user_input)
