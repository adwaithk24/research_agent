import uuid
import datetime
import requests
import json

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Miriel", page_icon="💬", layout="wide")

with open("./auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

if "column_data" not in st.session_state:
    st.session_state.column_data = []


if "chats" not in st.session_state:
    st.session_state.chats = {}

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

if "new_chat_name" not in st.session_state:
    st.session_state.new_chat_name = ""


# def login():
#     # TODO: IMPLEMENT AUTHENTICATION
#     st.session_state.authenticated = True
#     st.rerun()


def upload_pdf_to_backend(file, chat_id):
    try:
        response = requests.post(
            "http://localhost:8000/upload_pdf",
            files={"file": (file.name, file, "application/pdf")},
        )
        if response.status_code == 201:
            response_data = response.json()
            # Store the PDF ID and update chat status
            st.session_state.chats[chat_id].update(
                {
                    "pdf_id": response_data["pdf_id"],
                    "has_pdf": True,
                    "pdf_name": file.name,
                }
            )
            return True
        return False
    except Exception as e:
        st.error(f"Error uploading PDF: {str(e)}")
        return False


def create_new_chat():
    if not st.session_state.new_chat_name:
        return

    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "name": st.session_state.new_chat_name,
        "created_at": timestamp,
        "has_pdf": False,  # Track if PDF is uploaded
        "summary_generated": False,  # Track if summary has been generated
    }

    st.session_state.current_chat_id = chat_id
    st.session_state.messages[chat_id] = []
    st.session_state.messages[chat_id].append(
        {
            "role": "system",
            "content": "Please upload a PDF document to begin the conversation.",
        }
    )

    # Clear the input field
    st.session_state.new_chat_name = ""


def select_chat(chat_id):
    st.session_state.current_chat_id = chat_id


def send_message():
    if not st.session_state.user_input or not st.session_state.current_chat_id:
        return

    chat_id = st.session_state.current_chat_id
    user_message = st.session_state.user_input
    pdf_id = st.session_state.chats[chat_id].get("pdf_id")

    if not pdf_id:
        st.error("No PDF associated with this chat. Please upload a PDF first.")
        return

    # Add user message to chat
    st.session_state.messages[chat_id].append({"role": "user", "content": user_message})

    try:
        # Send question to backend with PDF ID
        response = requests.post(
            "http://localhost:8000/ask_question",
            json={
                "pdf_id": pdf_id,
                "question": user_message,
                "max_tokens": 500,  # You can adjust this value
            },
        )

        if response.status_code == 200:
            answer = response.json().get(
                "answer", "Sorry, I couldn't process your question."
            )
            st.session_state.messages[chat_id].append(
                {"role": "assistant", "content": answer}
            )
        else:
            st.session_state.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your question.",
                }
            )
    except Exception as e:
        st.session_state.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error processing your question: {str(e)}",
            }
        )

    # Clear the input
    st.session_state.user_input = ""


def insert_column_data(data_item):
    if not st.session_state.current_chat_id:
        return

    chat_id = st.session_state.current_chat_id

    st.session_state.messages[chat_id].append(
        {"role": "user", "content": f"Tell me about: {data_item}"}
    )

    bot_response = (
        f"Here's information about {data_item}. This is a placeholder response."
    )
    st.session_state.messages[chat_id].append(
        {"role": "assistant", "content": bot_response}
    )


def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
    if chat_id in st.session_state.messages:
        del st.session_state.messages[chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None
    st.rerun()


def clear_all_data():
    st.session_state.chats = {}
    st.session_state.messages = {}
    st.session_state.current_chat_id = None
    st.rerun()


def generate_summary(chat_id):
    pdf_id = st.session_state.chats[chat_id].get("pdf_id")

    if not pdf_id:
        st.error("No PDF associated with this chat. Please upload a PDF first.")
        return

    try:
        response = requests.post(
            "http://localhost:8000/summarize/",
            json={"pdf_id": pdf_id, "summary_length": 200},  # Default summary length
        )

        if response.status_code == 200:
            summary = response.json().get(
                "summary", "Sorry, I couldn't generate a summary."
            )
            st.session_state.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": f"Here's a summary of the document:\n\n{summary}",
                }
            )
            # Mark summary as generated
            # st.session_state.chats[chat_id]["summary_generated"] = True
        else:
            st.session_state.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error generating the summary.",
                }
            )
    except Exception as e:
        st.session_state.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error generating summary: {str(e)}",
            }
        )
    st.session_state.chats[chat_id]["summary_generated"] = True


# def render_main_app(authenticator: stauth.Authenticate):
authenticator.logout()
left_col, middle_col = st.columns([20, 80])

with left_col:
    st.header("Create New Chat")

    st.text_input(
        "Chat Name",
        key="new_chat_name",
        on_change=create_new_chat,
    )

    st.divider()
    st.subheader("Your Chats")

    if not st.session_state.chats:
        st.info("No chats yet. Create a new chat to get started!")

    for chat_id, chat in st.session_state.chats.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(
                f"{chat['name']}",
                key=f"chat_{chat_id}",
                use_container_width=True,
            ):
                select_chat(chat_id)
        with col2:
            if st.button("🗑️", key=f"delete_{chat_id}"):
                delete_chat(chat_id)

    if st.session_state.chats:
        st.divider()
        if st.button("Clear All Chats", type="secondary"):
            clear_all_data()

with middle_col:
    st.header("Chat")

    chat_container = st.container(height=500, border=True)

    with chat_container:
        if st.session_state.current_chat_id:
            chat_id = st.session_state.current_chat_id
            chat = st.session_state.chats[chat_id]
            user_input = st.session_state["user_input"] = ""

            st.subheader(f"Chat: {chat['name']}")

            # Show current PDF info if one is uploaded
            if chat.get("has_pdf", False):
                st.info(
                    f"📄 Current PDF: {chat.get('pdf_name')} (ID: {chat.get('pdf_id')})"
                )

            # Show PDF upload prompt if no PDF is uploaded yet
            if not chat.get("has_pdf", False):
                with st.status("Upload a PDF document", expanded=True) as upload_status:
                    uploaded_file = st.file_uploader(
                        "Upload a PDF document",
                        type=["pdf"],
                        key=f"{chat}.has_pdf",
                    )
                    if uploaded_file is not None:
                        if upload_pdf_to_backend(uploaded_file, chat_id):
                            st.session_state.messages[chat_id].append(
                                {
                                    "role": "system",
                                    "content": f"PDF '{uploaded_file.name}' has been uploaded and processed. You can now ask questions about the document.",
                                }
                            )
                            upload_status.update(
                                label="PDF uploaded successfully!",
                                state="complete",
                                expanded=False,
                            )
                            st.rerun()
                        else:
                            upload_status.update(
                                label="Failed to upload PDF",
                                state="error",
                            )
            else:

                # Show messages
                for message in st.session_state.messages.get(chat_id, []):
                    if message["role"] == "user":
                        st.chat_message("user").write(message["content"])
                    elif message["role"] == "system":
                        st.chat_message("system").write(message["content"])
                    else:
                        st.chat_message("assistant").write(message["content"])

                if st.button(
                    "Summarize Document",
                    type="primary",
                    disabled=chat.get("summary_generated", False),
                ):
                    generate_summary(chat_id)
                    st.rerun()
                # Show chat input box if PDF is uploaded
                st.text_input(
                    "Ask a question about the document",
                    key="user_input",
                    on_change=send_message,
                )
        else:
            st.markdown("### Create or select a chat to continue")
            st.markdown(
                "👈 Use the left panel to create a new chat or select an existing one"
            )

# with right_col:
#     st.header("Data Selection")

#     st.write("Select data to insert into chat:")

#     for data_item in st.session_state.column_data:
#         if st.button(data_item, key=f"data_{data_item}", use_container_width=True):
#             insert_column_data(data_item)
