import os

import requests
import streamlit as st

OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

st.set_page_config(page_title="OpenAI Wrapper", page_icon="OpenAI Wrapper", layout="centered")

st.title("OpenAI API Wrapper")
st.write("Use this like a normal chat. Add an OpenAI API key in the sidebar to start.")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi. Send a message and I’ll reply like a normal chat."}
    ]

with st.sidebar:
    st.header("Connection")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        value=st.session_state.openai_api_key,
        placeholder="sk-...",
    )
    st.session_state.openai_api_key = api_key.strip()

    model = st.text_input("Model", value=DEFAULT_MODEL)
    test_key = st.button("Validate key")

    if st.button("Clear chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat cleared. Send a new message when you're ready."}
        ]
        st.rerun()


def call_openai(prompt: str, key: str, selected_model: str) -> str:
    response = requests.post(
        f"{OPENAI_BASE_URL}/responses",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json={
            "model": selected_model,
            "input": prompt,
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()

    output_text = payload.get("output_text")
    if output_text:
        return output_text

    outputs = payload.get("output", [])
    collected = []
    for item in outputs:
        for content in item.get("content", []):
            text = content.get("text")
            if text:
                collected.append(text)
    return "\n".join(collected).strip() or str(payload)


def build_chat_prompt(messages: list[dict[str, str]]) -> str:
    transcript = [
        "You are a helpful chat assistant. Reply naturally and keep the conversation going."
    ]
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        transcript.append(f"{role.title()}: {content}")
    transcript.append("Assistant:")
    return "\n".join(transcript)


def validate_key(key: str) -> tuple[bool, str]:
    response = requests.get(
        f"{OPENAI_BASE_URL}/models",
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
    )
    if response.ok:
        return True, "Key validated successfully."
    try:
        detail = response.json().get("error", {}).get("message", response.text)
    except ValueError:
        detail = response.text
    return False, f"Key validation failed: {detail}"


if test_key:
    if not st.session_state.openai_api_key:
        st.error("Enter an API key first.")
    else:
        is_valid, message = validate_key(st.session_state.openai_api_key)
        if is_valid:
            st.success(message)
        else:
            st.error(message)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Send a message")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if not st.session_state.openai_api_key:
                    st.error("Add an OpenAI API key in the sidebar.")
                    st.stop()

                chat_prompt = build_chat_prompt(st.session_state.messages)
                result = call_openai(chat_prompt, st.session_state.openai_api_key, model.strip() or DEFAULT_MODEL)

                st.write(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
            except requests.HTTPError as exc:
                response = exc.response
                if response is None:
                    error_detail = str(exc)
                else:
                    try:
                        error_detail = response.json().get("error", {}).get("message", response.text)
                    except ValueError:
                        error_detail = response.text
                st.error(f"OpenAI request failed: {error_detail}")
            except requests.RequestException as exc:
                st.error(f"Network error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
