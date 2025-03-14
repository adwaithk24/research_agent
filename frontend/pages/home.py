import uuid
import datetime

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
    st.session_state.column_data = [
        "Customer Information",
        "Product Details",
        "Order History",
        "Payment Methods",
        "Shipping Options",
        "Return Policy",
        "Frequently Asked Questions",
        "Technical Support",
        "Account Settings",
    ]

# if "authenticated" not in st.session_state:
#     st.session_state.authenticated = False

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


def create_new_chat():

    chat_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.session_state.chats[chat_id] = {
        "id": chat_id,
        "name": st.session_state.new_chat_name,
        "created_at": timestamp,
    }

    st.session_state.current_chat_id = chat_id
    st.session_state.messages[chat_id] = []

    st.session_state.new_chat_name = ""


def select_chat(chat_id):
    st.session_state.current_chat_id = chat_id


def send_message():
    if not st.session_state.user_input or not st.session_state.current_chat_id:
        return

    user_message = st.session_state.user_input
    chat_id = st.session_state.current_chat_id

    st.session_state.messages[chat_id].append({"role": "user", "content": user_message})

    bot_response = f"Echo: {user_message}"
    st.session_state.messages[chat_id].append(
        {"role": "assistant", "content": bot_response}
    )

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


# def render_main_app(authenticator: stauth.Authenticate):
authenticator.logout()
left_col, middle_col, right_col = st.columns([20, 40, 20])

with left_col:
    st.button("Logout", on_click=authenticator.logout)
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

            st.subheader(f"Chat: {chat['name']}")

            for message in st.session_state.messages.get(chat_id, []):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        else:
            st.markdown("### Create or select a chat to continue")
            st.markdown(
                "👈 Use the left panel to create a new chat or select an existing one"
            )

    if st.session_state.current_chat_id:
        st.text_input("Type your message", key="user_input", on_change=send_message)
    else:
        st.text_input("Type your message", key="user_input", disabled=True)

with right_col:
    st.header("Data Selection")

    st.write("Select data to insert into chat:")

    for data_item in st.session_state.column_data:
        if st.button(data_item, key=f"data_{data_item}", use_container_width=True):
            insert_column_data(data_item)
