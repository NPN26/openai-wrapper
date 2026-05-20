# OpenAI Streamlit Wrapper

A minimal Streamlit app that accepts a user-supplied OpenAI API key, keeps it in session state, and sends prompts to the OpenAI Responses API from the server side.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The API key is not hardcoded or written to disk.
- The key is stored only in `st.session_state` for the current browser session.
- The app validates the key with `GET /v1/models` and sends prompts with `POST /v1/responses`.
