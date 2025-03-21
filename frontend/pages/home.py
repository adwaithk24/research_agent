import uuid
import datetime
import requests
import json
import logging

import streamlit as st
from streamlit import session_state as ss
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Miriel", page_icon="💬", layout="wide")

with open("./auth.yaml") as file:
    config = yaml.load(file, Loader=SafeLoader)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

if "column_data" not in ss:
    ss.column_data = []


if "chats" not in ss:
    ss.chats = {}

if "messages" not in ss:
    ss.messages = {}

if "current_chat_id" not in ss:
    ss.current_chat_id = None

if "new_chat_name" not in ss:
    ss.new_chat_name = ""

if "input_tokens" not in ss:
    ss.input_tokens = 0

if "output_tokens" not in ss:
    ss.output_tokens = 0

if "cost" not in ss:
    ss.cost = 0


ss.chats["1"] = {
    "id": "1",
    "name": "nvidia",
    "created_at": "2025-03-22 12:00:00",
    "has_pdf": True,
    "pdf_id": "1",
    "pdf_name": "nvidia.pdf",
}


def upload_pdf_to_backend(file, chat_id, ocr_method: str):
    try:
        logger.info(f"Uploading PDF to backend with OCR method: {ocr_method}")
        response = requests.post(
            "http://localhost:8000/upload_pdf",
            files={"file": (file.name, file, "application/pdf")},
            params={
                "parser": ocr_method,
                "chunking_strategy": chunking_strategy,
                "vector_store": vector_store,
            },
        )
        if response.status_code == 201:
            response_data = response.json()
            # Store the PDF ID and update chat status
            ss.chats[chat_id].update(
                {
                    "pdf_id": response_data["pdf_id"],
                    "has_pdf": True,
                    "pdf_name": file.name,
                }
            )
            logger.info(f"PDF uploaded to backend: {response_data['pdf_id']}")
            return True
        return False
    except Exception as e:
        logger.error(f"Error uploading PDF to backend: {str(e)}")
        st.error("Error uploading PDF to backend")
        return False


def create_new_chat():
    if not ss.new_chat_name:
        return

    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ss.chats[chat_id] = {
        "id": chat_id,
        "name": ss.new_chat_name,
        "created_at": timestamp,
        "has_pdf": False,  # Track if PDF is uploaded
        "summary_generated": False,  # Track if summary has been generated
    }
    logger.info(f"Created new chat: {ss.chats[chat_id]}")
    ss.current_chat_id = chat_id
    ss.messages[chat_id] = []
    ss.messages[chat_id].append(
        {
            "role": "system",
            "content": "Please upload a PDF document to begin the conversation.",
        }
    )

    # Clear the input field
    ss.new_chat_name = ""


def select_chat(chat_id):
    ss.current_chat_id = chat_id


def send_message():
    if not ss.user_input or not ss.current_chat_id:
        return

    chat_id = ss.current_chat_id
    user_message = ss.user_input
    pdf_id = ss.chats[chat_id].get("pdf_id")
    logger.info(
        f"Creating new message for chat with id: {chat_id}, user message: {user_message}, pdf_id: {pdf_id}"
    )
    if not pdf_id:
        logger.error("No PDF associated with this chat. Please upload a PDF first.")
        st.error("No PDF associated with this chat. Please upload a PDF first.")
        return

    # Add user message to chat
    ss.messages[chat_id].append({"role": "user", "content": user_message})

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
            response_data = response.json()
            answer = response_data.get(
                "answer", "Sorry, I couldn't process your question."
            )

            # Display token usage and cost metrics
            usage_metrics = response_data.get("usage_metrics", {})
            if usage_metrics:
                ss.input_tokens += usage_metrics.get("input_tokens", 0)
                ss.output_tokens += usage_metrics.get("output_tokens", 0)
                ss.cost += usage_metrics.get("cost", 0)

            ss.messages[chat_id].append({"role": "assistant", "content": answer})
            logger.info(f"Answer received from backend: {answer}")
        else:
            ss.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error processing your question.",
                }
            )
            logger.error(
                f"Error processing your question: {response.status_code}, {response.text}"
            )
    except Exception as e:
        logger.error(f"Error processing your question: {str(e)}")
        ss.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error processing your question: {str(e)}",
            }
        )

    # Clear the input
    ss.user_input = ""


def insert_column_data(data_item):
    if not ss.current_chat_id:
        return

    chat_id = ss.current_chat_id

    ss.messages[chat_id].append(
        {"role": "user", "content": f"Tell me about: {data_item}"}
    )

    bot_response = (
        f"Here's information about {data_item}. This is a placeholder response."
    )
    ss.messages[chat_id].append({"role": "assistant", "content": bot_response})


def delete_chat(chat_id):
    logger.info(f"Deleting chat with id: {chat_id}")
    if chat_id in ss.chats:
        del ss.chats[chat_id]
    if chat_id in ss.messages:
        del ss.messages[chat_id]
    if ss.current_chat_id == chat_id:
        ss.current_chat_id = None
    st.rerun()


def clear_all_data():
    logger.info("Clearing all chat data")
    ss.chats = {}
    ss.messages = {}
    ss.current_chat_id = None
    st.rerun()


def generate_summary(chat_id):
    pdf_id = ss.chats[chat_id].get("pdf_id")

    if not pdf_id:
        logger.error("No PDF associated with this chat. Please upload a PDF first.")
        st.error("No PDF associated with this chat. Please upload a PDF first.")
        return

    try:
        response = requests.post(
            "http://localhost:8000/ask_question",
            json={
                "pdf_id": pdf_id,
                "question": "Summarize the document",
                "max_tokens": 500,  # You can adjust this value
            },
        )
        logger.info(f"Summary response: {response.json()}")
        if response.status_code == 200:
            response_data = response.json()
            summary = response_data.get(
                "answer", "Sorry, I couldn't process your question."
            )

            ss.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": f"Here's a summary of the document:\n\n{summary}",
                }
            )
            logger.info(f"Summary generated: {summary}")
            usage_metrics = response_data.get("usage_metrics", {})
            if usage_metrics:
                ss.input_tokens += usage_metrics.get("input_tokens", 0)
                ss.output_tokens += usage_metrics.get("output_tokens", 0)
                ss.cost += usage_metrics.get("cost", 0)
            # Mark summary as generated
            # st.session_state.chats[chat_id]["summary_generated"] = True
        else:
            ss.messages[chat_id].append(
                {
                    "role": "assistant",
                    "content": "Sorry, I encountered an error generating the summary.",
                }
            )
            logger.error(
                f"Error generating summary: {response.status_code}, {response.text}"
            )
    except Exception as e:
        ss.messages[chat_id].append(
            {
                "role": "assistant",
                "content": f"Error generating summary: {str(e)}",
            }
        )
        logger.error(f"Error generating summary: {str(e)}")
    ss.chats[chat_id]["summary_generated"] = True


# def render_main_app(authenticator: stauth.Authenticate):
authenticator.logout()
left_col, middle_col, right_col = st.columns([20, 60, 20])

with left_col:
    st.header("Create New Chat")

    st.text_input(
        "Chat Name",
        key="new_chat_name",
        on_change=create_new_chat,
    )

    st.divider()
    st.subheader("Your Chats")

    if not ss.chats:
        st.info("No chats yet. Create a new chat to get started!")

    for chat_id, chat in ss.chats.items():
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

    if ss.chats:
        st.divider()
        if st.button("Clear All Chats", type="secondary"):
            clear_all_data()

with middle_col:
    st.header("Chat")

    chat_container = st.container(height=500, border=True)

    with chat_container:
        if ss.current_chat_id:
            chat_id = ss.current_chat_id
            chat = ss.chats[chat_id]
            user_input = ss["user_input"] = ""

            st.subheader(f"Chat: {chat['name']}")

            # Show PDF upload prompt if no PDF is uploaded yet
            if not chat.get("has_pdf", False):
                with st.status("Upload a PDF document", expanded=True) as upload_status:
                    ocr_method = st.selectbox(
                        "Select parsing method",
                        options=["mistral", "docling"],
                        index=1,
                    )
                    logger.info(f"Selected OCR method: {ocr_method}")
                    chunking_strategy = st.selectbox(
                        "Select chunking strategy",
                        options=["fixed-size", "recursive", "semantic"],
                        index=1,
                    )
                    logger.info(f"Selected chunking strategy: {chunking_strategy}")
                    vector_store = st.selectbox(
                        "Select vector store",
                        options=["pinecone", "chroma", "naive"],
                        index=1,
                    )
                    logger.info(f"Selected vector store: {vector_store}")

                    uploaded_file = st.file_uploader(
                        "Upload a PDF document",
                        type=["pdf"],
                    )
                    if uploaded_file is not None:
                        if upload_pdf_to_backend(uploaded_file, chat_id, ocr_method):
                            ss.messages[chat_id].append(
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
                            ss.chats[chat_id]["has_pdf"] = True
                            st.rerun()
                        else:
                            upload_status.update(
                                label="Failed to upload PDF",
                                state="error",
                            )
                            ss.chats[chat_id]["has_pdf"] = False
            else:

                st.info(
                    f"📄 Current PDF: {chat.get('pdf_name')} (ID: {chat.get('pdf_id')})"
                )

                # Show messages
                for message in ss.messages.get(chat_id, []):
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

with right_col:
    st.header("Usage Metrics")
    with st.expander("📊 Summary Token Usage & Cost", expanded=True):
        st.metric("Input Tokens", ss.get("input_tokens", "N/A"))
        st.metric("Output Tokens", ss.get("output_tokens", "N/A"))
        st.metric("Cost ($)", f"${ss.get('cost', 0):.4f}")
