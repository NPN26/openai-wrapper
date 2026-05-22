import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import streamlit as st

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

def build_chat_model(api_key: str, model_name: str):
    return init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=api_key,
        base_url=OPENAI_BASE_URL,
        temperature=0.5,
        timeout=300,
        max_tokens=25000,
    )


def to_langchain_messages(messages: list[dict[str, str]]):
    converted = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "assistant":
            converted.append(AIMessage(content=content))
        elif role == "system":
            converted.append(SystemMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


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

def add_to_temp_file(content: str, file_path: str):
    with open(file_path, "a", encoding="utf-8") as tmp:
        tmp.write(content + "\n")


def read_temp_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, "r", encoding="utf-8") as tmp:
        return tmp.read()


def clear_temp_file(file_path: str):
    with open(file_path, "w", encoding="utf-8"):
        pass


def persist_messages_to_temp_file(messages: list[dict[str, str]], file_path: str):
    clear_temp_file(file_path)
    for message in messages:
        add_to_temp_file(f"{message.get('role', 'user')}: {message.get('content', '')}", file_path)

st.set_page_config(page_title="OpenAI Wrapper", page_icon="OpenAI Wrapper", layout="centered")

st.title("OpenAI API Wrapper")
st.write("Use this like a normal chat. Add an OpenAI API key in the sidebar to start.")

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

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. Send a message and I’ll reply like a normal chat."}
    ]
 
if "file_path" not in st.session_state:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.close()
    st.session_state.file_path = tmp.name
    persist_messages_to_temp_file(st.session_state.messages, st.session_state.file_path)

with st.sidebar:
    st.header("Connection")
    st.subheader("Endpoint")
    base_url = st.text_input(
        "OpenAI Base URL",
        value=st.session_state.openai_base_url,
        help="Custom OpenAI API base URL (e.g. https://api.openai.com/v1 or an enterprise endpoint).",
    )
    st.session_state.openai_base_url = base_url.strip() or OPENAI_BASE_URL
    OPENAI_BASE_URL = st.session_state.openai_base_url
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-...",
    )
    st.session_state.openai_api_key = api_key.strip()

    if st.session_state.openai_api_key != st.session_state.validated_api_key:
        st.session_state.api_key_valid = False

    test_key = st.button("Validate key")

    st.session_state.selected_model = st.text_input(
        "Model / deployment",
        value=st.session_state.selected_model,
        help="For Azure endpoints, enter the deployment name, not the underlying model name.",
    )

    if st.session_state.api_key_valid:
        st.caption("Key validated against the configured endpoint.")

    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Send a new message when you're ready."}
        ]
        clear_temp_file(st.session_state.file_path)
        persist_messages_to_temp_file(st.session_state.messages, st.session_state.file_path)
        st.rerun()

    st.subheader("Session Temp File")
    st.caption(st.session_state.file_path)
    if st.button("Show temp file content"):
        file_content = read_temp_file(st.session_state.file_path)
        if file_content:
            st.text_area("Transcript from temp file", value=file_content, height=220)
        else:
            st.info("Temp file is currently empty.")

def validate_key(key: str) -> tuple[bool, str]:
    try:
        model = build_chat_model(key, st.session_state.selected_model or DEFAULT_MODEL)
        model.invoke(
            [
                SystemMessage(content="You are a helpful chat assistant. Reply naturally and keep the conversation going."),
                HumanMessage(content="ping"),
            ]
        )
        return True, "Key validated successfully."
    except Exception as exc:
        detail = str(exc)
        if "Malformed identifier" in detail:
            detail = (
                f"{detail} Use the Azure deployment name in the Model field and keep the base URL as the resource endpoint, "
                "for example https://YOUR-RESOURCE.openai.azure.com/openai/v1/."
            )
        return False, f"Key validation failed: {detail}"


def generate_reply(api_key: str, model_name: str, messages: list[dict[str, str]]) -> str:
    model = build_chat_model(api_key, model_name)
    response = model.invoke(
        [
            SystemMessage(content="You are a helpful chat assistant. Reply naturally and keep the conversation going."),
            *to_langchain_messages(messages),
        ]
    )
    return response_to_text(response) or "No response from model."


if test_key:
    if not st.session_state.openai_api_key:
        st.error("Enter an API key first.")
    else:
        is_valid, message = validate_key(st.session_state.openai_api_key)
        if is_valid:
            st.session_state.validated_api_key = st.session_state.openai_api_key
            st.session_state.api_key_valid = True
            st.success(message)
        else:
            st.session_state.validated_api_key = ""
            st.session_state.api_key_valid = False
            st.error(message)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Send a message")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    add_to_temp_file(f"user: {prompt}", st.session_state.file_path)

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if not st.session_state.openai_api_key:
                    st.error("Add an OpenAI API key in the sidebar.")
                    st.stop()

                result = generate_reply(
                    api_key=st.session_state.openai_api_key,
                    model_name=st.session_state.selected_model or DEFAULT_MODEL,
                    messages=st.session_state.messages + [{"role": "user", "content": prompt}],
                )

                st.write(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
                add_to_temp_file(f"assistant: {result}", st.session_state.file_path)
            except Exception as exc:
                error_detail = str(exc)
                if "Malformed identifier" in error_detail:
                    error_detail = (
                        f"{error_detail} Use the Azure deployment name in the Model field, not the underlying model name."
                    )
                st.error(f"OpenAI request failed: {error_detail}")
