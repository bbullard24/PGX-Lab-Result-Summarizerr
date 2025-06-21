import os
import base64
import re
import json
import uuid
from typing import List

import streamlit as st
import openai
from openai import AssistantEventHandler
from tools import TOOL_MAP
from typing_extensions import override
from dotenv import load_dotenv
import streamlit_authenticator as stauth

load_dotenv()

# Initialize session state variables
def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = []
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []
    if "in_progress" not in st.session_state:
        st.session_state.in_progress = False
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = {}

init_session_state()

def str_to_bool(str_input):
    return isinstance(str_input, str) and str_input.lower() == "true"

# Initialize OpenAI client
def init_openai_client():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    azure_openai_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.environ.get("AZURE_OPENAI_KEY")
    
    if azure_openai_endpoint and azure_openai_key:
        return openai.AzureOpenAI(
            api_key=azure_openai_key,
            api_version="2024-05-01-preview",
            azure_endpoint=azure_openai_endpoint,
        )
    return openai.OpenAI(api_key=openai_api_key)

client = init_openai_client()

# Authentication setup
def setup_authentication():
    authentication_required = str_to_bool(os.environ.get("AUTHENTICATION_REQUIRED", False))
    if not authentication_required:
        return None
    
    if "credentials" in st.secrets:
        return stauth.Authenticate(
            st.secrets["credentials"].to_dict(),
            st.secrets["cookie"]["name"],
            st.secrets["cookie"]["key"],
            st.secrets["cookie"]["expiry_days"],
        )
    return None

authenticator = setup_authentication()

class EventHandler(AssistantEventHandler):
    @override
    def on_event(self, event): pass

    @override
    def on_text_created(self, text):
        st.session_state.current_message = ""
        with st.chat_message("Assistant"):
            st.session_state.current_markdown = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        if snapshot.value:
            text_value = re.sub(r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", "Download Link", snapshot.value)
            st.session_state.current_message = text_value
            st.session_state.current_markdown.markdown(text_value, True)

    @override
    def on_text_done(self, text):
        format_text = format_annotation(text)
        st.session_state.current_markdown.markdown(format_text, True)
        st.session_state.chat_log.append({"name": "assistant", "msg": format_text})

    @override
    def on_tool_call_created(self, tool_call):
        if tool_call.type == "code_interpreter":
            st.session_state.current_tool_input = ""
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()

    @override
    def on_tool_call_delta(self, delta, snapshot):
        if 'current_tool_input_markdown' not in st.session_state:
            with st.chat_message("Assistant"):
                st.session_state.current_tool_input_markdown = st.empty()
        if delta.type == "code_interpreter":
            if delta.code_interpreter.input:
                st.session_state.current_tool_input += delta.code_interpreter.input
                input_code = f"### code interpreter\ninput:\n```python\n{st.session_state.current_tool_input}\n```"
                st.session_state.current_tool_input_markdown.markdown(input_code, True)

    @override
    def on_tool_call_done(self, tool_call):
        st.session_state.tool_calls.append(tool_call)
        if tool_call.type == "code_interpreter":
            input_code = f"### code interpreter\ninput:\n```python\n{tool_call.code_interpreter.input}\n```"
            st.session_state.current_tool_input_markdown.markdown(input_code, True)
            st.session_state.chat_log.append({"name": "assistant", "msg": input_code})
            for output in tool_call.code_interpreter.outputs:
                if output.type == "logs":
                    output_msg = f"### code interpreter\noutput:\n```\n{output.logs}\n```"
                    with st.chat_message("Assistant"):
                        st.markdown(output_msg, True)
                        st.session_state.chat_log.append({"name": "assistant", "msg": output_msg})

def create_thread():
    return client.beta.threads.create()

def create_message(thread, content, files: List = None):
    attachments = []
    if files:
        for file in files:
            attachments.append({
                "file_id": file.id,
                "tools": [{"type": "code_interpreter"}, {"type": "file_search"}]
            })
    client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=content, attachments=attachments
    )

def handle_uploaded_files(uploaded_files):
    return [client.files.create(file=file, purpose="assistants") for file in uploaded_files]

def format_annotation(text):
    citations = []
    text_value = text.value
    for index, annotation in enumerate(text.annotations):
        text_value = text_value.replace(annotation.text, f" [{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {file_citation.quote} from {cited_file.filename}")
        elif file_path := getattr(annotation, "file_path", None):
            link_tag = create_file_link(annotation.text.split("/")[-1], file_path.file_id)
            text_value = re.sub(r"\[(.*?)\]\s*\(\s*(.*?)\s*\)", link_tag, text_value)
    text_value += "\n\n" + "\n".join(citations)
    return text_value

def create_file_link(file_name, file_id):
    content = client.files.content(file_id)
    content_type = content.response.headers["content-type"]
    b64 = base64.b64encode(content.text.encode(content.encoding)).decode()
    return f'<a href="data:{content_type};base64,{b64}" download="{file_name}">Download Link</a>'

def run_stream(user_input, files: List, selected_assistant_id):
    if "thread" not in st.session_state:
        st.session_state.thread = create_thread()
    create_message(st.session_state.thread, user_input, files)
    with client.beta.threads.runs.stream(
        thread_id=st.session_state.thread.id,
        assistant_id=selected_assistant_id,
        event_handler=EventHandler(),
    ) as stream:
        stream.until_done()

def process_files(uploaded_files, user_msg, assistant_id):
    if not uploaded_files:
        return
    
    # Process each file in sequence
    for file in uploaded_files:
        file_key = f"{file.name}_{file.size}"
        
        # Skip if file was already processed
        if file_key in st.session_state.processed_files:
            continue
            
        with st.expander(f"Processing: {file.name}", expanded=True):
            st.write(f"**Processing PDF:** {file.name}")
            
            # Upload and process the file
            try:
                openai_files = handle_uploaded_files([file])
                run_stream(user_msg, openai_files, assistant_id)
                st.session_state.processed_files[file_key] = True
                st.success(f"Completed processing: {file.name}")
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")

def render_chat():
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.markdown(chat["msg"], True)

def reset_chat():
    st.session_state.chat_log = []
    st.session_state.in_progress = False

def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.chat_log = []
    st.session_state.processed_files = {}
    st.session_state.uploader_key += 1
    st.rerun()

def load_chat_screen(assistant_id, assistant_title):
    uploaded_files = st.sidebar.file_uploader(
        os.environ.get("ENABLED_FILE_UPLOAD_MESSAGE", "Upload PDF files"),
        type=["pdf"],
        disabled=st.session_state.in_progress,
        key=f"file_uploader_{st.session_state.uploader_key}",
        accept_multiple_files=True
    )
    
    st.title(assistant_title if assistant_title else "")
    
    user_msg = st.chat_input(
        "Message", 
        on_submit=disable_form, 
        disabled=st.session_state.in_progress
    )
    
    if user_msg:
        render_chat()
        with st.chat_message("user"):
            st.markdown(user_msg, True)
        st.session_state.chat_log.append({"name": "user", "msg": user_msg})
        
        if uploaded_files:
            process_files(uploaded_files, user_msg, assistant_id)
        else:
            run_stream(user_msg, None, assistant_id)
            
        st.session_state.in_progress = False
        st.rerun()
    
    render_chat()

def disable_form():
    st.session_state.in_progress = True

def main():
    if authenticator and not st.session_state.get("authentication_status"):
        authenticator.login()
        if not st.session_state["authentication_status"]:
            st.error("Username/password is incorrect")
            return
    
    multi_agents = os.environ.get("OPENAI_ASSISTANTS", None)
    single_agent_id = os.environ.get("ASSISTANT_ID", None)
    single_agent_title = os.environ.get("ASSISTANT_TITLE", "Assistants API UI")

    with st.sidebar:
        if authenticator and st.session_state.get("authentication_status"):
            authenticator.logout()
            
        if st.button("➕ Start New Chat"):
            start_new_chat()

    if multi_agents:
        assistants_json = json.loads(multi_agents)
        assistants_object = {obj["title"]: obj for obj in assistants_json}
        selected_assistant = st.sidebar.selectbox(
            "Select an assistant profile?",
            list(assistants_object.keys()),
            index=None,
            placeholder="Select an assistant profile...",
            on_change=reset_chat,
        )
        if selected_assistant:
            load_chat_screen(
                assistants_object[selected_assistant]["id"],
                assistants_object[selected_assistant]["title"],
            )
    elif single_agent_id:
        load_chat_screen(single_agent_id, single_agent_title)

if __name__ == "__main__":
    main()