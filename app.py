import os
from pathlib import Path
from uuid import uuid4
import requests
from typing import TypedDict, Annotated, cast, Any
import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from psycopg import Connection
import streamlit as st

os.environ["LANGSMITH_TRACING"] = st.secrets["LANGSMITH_TRACING"]
os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
os.environ["LANGSMITH_ENDPOINT"] = st.secrets["LANGSMITH_ENDPOINT"]
OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1"
DB_URI = st.secrets["POSTGRES_URI"]
SYSTEM_PROMPT = "You are a helpful chat assistant. Reply naturally and keep the conversation going."

class State(TypedDict):
    messages: Annotated[list[AIMessage | HumanMessage], add_messages]
    
model = init_chat_model(
    temperature=0.5,
    timeout=300,
    max_tokens=25000,
    model_provider="openai",
    configurable_fields=["model", "api_key", "base_url"],
)

def call_model(state: State, config: RunnableConfig) -> State:
    response = model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"], config=config)
    return {"messages": [response]}
    
builder = StateGraph(State)
builder.add_node("agent", call_model)
builder.set_entry_point("agent")
builder.add_edge("agent",END)

def require_db_uri() -> str:
    if not DB_URI:
        raise ValueError("POSTGRES_URI is not set. Add it to your environment before starting the app.")
    return DB_URI

@st.cache_resource
def get_graph():
    try:
        conn = Connection.connect(require_db_uri(), autocommit=True)
        # Set connection timeout and keepalive
        conn.timeout = 30
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        return builder.compile(checkpointer=checkpointer)
    except Exception as e:
        st.error(f"Failed to connect to database: {str(e)}")
        st.stop()

def get_graph_with_retry():
    """Get graph with connection retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            graph = get_graph()
            # Test the connection
            graph.get_state({"configurable": {"thread_id": "test"}})
            return graph
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                st.cache_resource.clear()
            else:
                raise


def response_to_text(response) -> str:
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        collected = []
        for item in content:
            if isinstance(item, str):
                collected.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    collected.append(text)
        return "\n".join(collected).strip()
    return str(content).strip()

st.set_page_config(page_title="OpenAI Wrapper", page_icon="OpenAI Wrapper", layout="centered")

st.title("OpenAI API Wrapper")
st.write("Set your endpoint and API key below, validate once, then start chatting.")

if not DB_URI:
    st.error("POSTGRES_URI is not set. Configure your cloud Postgres connection string to enable persistent chat memory.")
    st.stop()

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "validated_api_key" not in st.session_state:
    st.session_state.validated_api_key = ""

if "api_key_valid" not in st.session_state:
    st.session_state.api_key_valid = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = DEFAULT_MODEL

if "openai_base_url" not in st.session_state:
    st.session_state.openai_base_url = OPENAI_BASE_URL

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid4())

if "thread_history" not in st.session_state:
    st.session_state.thread_history = {st.session_state.thread_id: "Chat 1"}

if "thread_order" not in st.session_state:
    st.session_state.thread_order = [st.session_state.thread_id]

if "show_connection_section" not in st.session_state:
    st.session_state.show_connection_section = True


def register_thread(thread_id: str) -> None:
    if thread_id not in st.session_state.thread_history:
        label = f"Chat {len(st.session_state.thread_history) + 1}"
        st.session_state.thread_history[thread_id] = label
        st.session_state.thread_order.insert(0, thread_id)

def check_api_key(api_key: str, base_url: str = OPENAI_BASE_URL) -> bool:
    try:
        if base_url != "https://api.openai.com/v1":
            response = requests.get(
                base_url,
                headers={"api-key": api_key}
            )
            return response.status_code != 401
        else:
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            return response.status_code != 401
    except requests.RequestException:
        return False

def validate_key(key: str) -> tuple[bool, str]:
    try:
        if check_api_key(key, st.session_state.openai_base_url or OPENAI_BASE_URL):
            return True, "Key is valid for the configured endpoint."
        else:
            return False, "Key is invalid for the configured endpoint."
    except Exception as exc:
        detail = str(exc)
        if "Malformed identifier" in detail:
            detail = (
                f"{detail} Use the Azure deployment name in the Model field and keep the base URL as the resource endpoint, "
                "for example https://YOUR-RESOURCE.openai.azure.com/openai/v1/."
            )
        return False, f"Key validation failed: {detail}"


if not st.session_state.api_key_valid:
    st.session_state.show_connection_section = True

test_key = False
if st.session_state.show_connection_section:
    st.subheader("Connection")
    st.session_state.openai_base_url = (
        st.text_input(
            "OpenAI Base URL",
            value=st.session_state.openai_base_url,
            help="Custom OpenAI API base URL (e.g. https://api.openai.com/v1 or an enterprise endpoint).",
        ).strip()
        or OPENAI_BASE_URL
    )

    st.session_state.openai_api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-...",
    ).strip()

    st.session_state.selected_model = st.text_input(
        "Model / deployment",
        value=st.session_state.selected_model,
        help="For Azure endpoints, enter the deployment name, not the underlying model name.",
    )

    if st.session_state.openai_api_key != st.session_state.validated_api_key:
        st.session_state.api_key_valid = False

    test_key = st.button("Validate key")
else:
    st.success("Key validated. You can start chatting.")
    if st.button("Change connection settings"):
        st.session_state.show_connection_section = True
        st.rerun()

if st.button("Start new chat"):
    st.session_state.thread_id = str(uuid4())
    register_thread(st.session_state.thread_id)
    st.rerun()

with st.sidebar:
    st.header("Past chats")
    register_thread(st.session_state.thread_id)
    selected_thread = st.radio(
        "Saved threads",
        options=st.session_state.thread_order,
        index=st.session_state.thread_order.index(st.session_state.thread_id),
        format_func=lambda thread: f"{st.session_state.thread_history.get(thread, 'Chat')} ({thread[:8]})",
    )

    if selected_thread != st.session_state.thread_id:
        st.session_state.thread_id = selected_thread
        st.rerun()

    st.caption(f"Current thread: {st.session_state.thread_id}")

if test_key:
    if not st.session_state.openai_api_key:
        st.error("Enter an API key first.")
    else:
        is_valid, message = validate_key(st.session_state.openai_api_key)
        if is_valid:
            st.session_state.validated_api_key = st.session_state.openai_api_key
            st.session_state.api_key_valid = True
            st.session_state.show_connection_section = False
            st.success(message)
            st.rerun()
        else:
            st.session_state.validated_api_key = ""
            st.session_state.api_key_valid = False
            st.error(message)
            
config: RunnableConfig = {
    "configurable": {
        "thread_id": st.session_state.thread_id,
        "model": st.session_state.selected_model or DEFAULT_MODEL,
        "api_key": st.session_state.openai_api_key,
        "base_url": st.session_state.openai_base_url or OPENAI_BASE_URL,
    }
}
try:
    graph = get_graph_with_retry()
except Exception as e:
    st.error(f"Could not establish database connection: {str(e)}")
    st.stop()

try:
    snapshot = graph.get_state(config)
    prior_messages = snapshot.values.get("messages", [])
except Exception:
    prior_messages = []
            
for msg in prior_messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(response_to_text(msg))
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(response_to_text(msg))

if not st.session_state.api_key_valid:
    st.info("Validate your API key to enable chat input.")

prompt = st.chat_input("Send a message", disabled=not st.session_state.api_key_valid)

if prompt:
    if not st.session_state.openai_api_key:
        st.error("Add an OpenAI API key in the connection section above.")
        st.stop()

    if not st.session_state.api_key_valid:
        st.error("Validate your API key before sending messages.")
        st.stop()

    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Retry logic for connection issues
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        result_state = graph.invoke(
                            {"messages": [HumanMessage(content=prompt)]},
                            config=config
                        )
                        st.write(response_to_text(result_state["messages"][-1]))
                        break
                    except Exception as e:
                        if "connection" in str(e).lower() and attempt < max_retries - 1:
                            st.cache_resource.clear()
                            graph = get_graph_with_retry()
                        else:
                            raise
            except Exception as exc:
                error_detail = str(exc)
                if "Malformed identifier" in error_detail:
                    error_detail += " Use the Azure deployment name in the Model field, not the underlying model name."
                elif "connection" in error_detail.lower():
                    error_detail = "Database connection lost. Please try again. If this persists, check your POSTGRES_URI."
                st.error(f"Request failed: {error_detail}")
