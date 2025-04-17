import streamlit as st
import os
from app_safe import PKTravelsChatbot
import uuid
import base64
from datetime import datetime

# Create images folder if it doesn't exist
IMAGES_FOLDER = "images"
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="PK Travels - Explore Pakistan",
    page_icon="🏔️",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .stApp {
        background-color: #f5f7f9;
    }
    .logo-container {
        text-align: center;
        padding: 1rem;
    }
    .title-container {
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .chat-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 50%;
        height: 50vh;
        z-index: 1000;
    }
    .chat-button {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #009900;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        z-index: 1001;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .chat-button:hover {
        transform: scale(1.1);
    }
    .chat-window {
        background-color: white;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 15px;
        width: 100%;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
    }
    .chat-messages {
        overflow-y: auto;
        flex-grow: 1;
        margin-bottom: 15px;
        padding-right: 8px;
    }
    .chat-input {
        border-top: 1px solid #e0e0e0;
        padding-top: 15px;
    }
    .user-message {
        background-color: #DCF8C6;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        text-align: right;
        margin-left: 50px;
    }
    .bot-message {
        background-color: #f1f1f1;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        margin-right: 50px;
    }
    .message-container {
        margin-bottom: 10px;
    }
    .travel-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .travel-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .featured-destinations {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .pakistan-colors {
        color: #009900;
    }
    /* Hide the default button */
    .stButton button[kind="primary"] {
        display: none !important;
    }
    @media (max-width: 768px) {
        .chat-container {
            width: 90%;
            height: 60vh;
        }
    }
    /* Additional style to ensure the button is visible and clickable */
    .floating-chat-btn {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background-color: #009900;
        display: flex;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        z-index: 9999;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: transform 0.3s ease;
    }
    .floating-chat-btn:hover {
        transform: scale(1.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get image path
def get_image_path(image_name, default_placeholder):
    image_path = os.path.join(IMAGES_FOLDER, image_name)
    if os.path.exists(image_path):
        return image_path
    return default_placeholder

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if 'chatbot' not in st.session_state:
    st.session_state.chatbot = PKTravelsChatbot()

if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False

# Function to toggle chat visibility
def toggle_chat():
    st.session_state.chat_visible = not st.session_state.chat_visible
    st.rerun()

# Function to handle sending a message
def send_message(message):
    if message:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": message})
        
        # Get response from chatbot
        response = st.session_state.chatbot.chat(message, st.session_state.session_id)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})
        
        # Rerun to update the UI
        st.rerun()

# Main layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Logo placeholder
    logo_path = get_image_path("logo.png", "https://placehold.co/300x150?text=PK+Travels+Logo")
    st.markdown(f"""
    <div class="logo-container">
        <img src="{logo_path}" alt="PK Travels Logo" width="300" style="border-radius: 10px;">
    </div>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("""
    <div class="title-container">
        <h1><span class="pakistan-colors">Welcome to PK Travels</span></h1>
        <p>Discover the beauty of Pakistan - From majestic mountains to historical treasures!</p>
    </div>
    """, unsafe_allow_html=True)

    # Featured destinations
    st.markdown('<h2><span class="pakistan-colors">Explore Pakistan\'s Treasures</span></h2>', unsafe_allow_html=True)

    # Pakistan-specific destinations with image handling
    destinations = [
        {"name": "Hunza Valley", "image": "hunza.jpg", "placeholder": "https://placehold.co/300x200?text=Hunza+Valley", "desc": "Experience breathtaking views of snow-capped peaks, lush green valleys, and the famous Attabad Lake in Northern Pakistan."},
        {"name": "Swat Valley", "image": "swat.jpg", "placeholder": "https://placehold.co/300x200?text=Swat+Valley", "desc": "Known as the 'Switzerland of Pakistan', featuring stunning landscapes, crystal-clear rivers, and green meadows."},
        {"name": "Skardu", "image": "skardu.jpg", "placeholder": "https://placehold.co/300x200?text=Skardu", "desc": "Gateway to the mighty Karakoram Range with beautiful Shangrila Resort and mystical Deosai Plains nearby."},
        {"name": "Fairy Meadows", "image": "fairy_meadows.jpg", "placeholder": "https://placehold.co/300x200?text=Fairy+Meadows", "desc": "Spectacular camping site offering panoramic views of Nanga Parbat, the world's ninth-highest mountain."},
        {"name": "Lahore", "image": "lahore.jpg", "placeholder": "https://placehold.co/300x200?text=Lahore", "desc": "Cultural heart of Pakistan with magnificent Mughal architecture, vibrant food scene, and historical monuments."},
        {"name": "Kalash Valley", "image": "kalash.jpg", "placeholder": "https://placehold.co/300x200?text=Kalash+Valley", "desc": "Home to the ancient Kalash tribe, known for their unique culture, colorful festivals, and traditional way of life."},
    ]

    # Display featured destinations
    st.markdown('<div class="featured-destinations">', unsafe_allow_html=True)
    for dest in destinations:
        image_src = get_image_path(dest["image"], dest["placeholder"])
        st.markdown(f"""
        <div class="travel-card">
            <img src="{image_src}" alt="{dest['name']}" width="100%">
            <h3>{dest['name']}</h3>
            <p>{dest['desc']}</p>
            <button style="background-color: #009900; color: white; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; transition: background-color 0.3s ease;">
                View Packages
            </button>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Hidden button that will be triggered by our custom chat button
chat_toggle_button = st.button("Open Chat", key="chat_toggle", on_click=toggle_chat)

# Use a different approach for the chat button
st.markdown("""
<div class="floating-chat-btn" onclick="toggleChat()">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white">
        <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
    </svg>
</div>

<script>
    // Function to find and click the Streamlit button
    function toggleChat() {
        // Use iframe approach to find elements inside Streamlit
        const allButtons = window.parent.document.querySelectorAll('button');
        
        // Look for our specific button
        for (const button of allButtons) {
            if (button.innerText.includes('Open Chat')) {
                button.click();
                return;
            }
        }
        
        // Alternative selector if the above doesn't work
        const primaryButtons = window.parent.document.querySelectorAll('button[kind="primary"]');
        if (primaryButtons.length > 0) {
            primaryButtons[0].click();
        }
    }
    
    // Expose the function so it can be called from onclick
    window.toggleChat = toggleChat;
</script>
""", unsafe_allow_html=True)

# Chat window
if st.session_state.chat_visible:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="chat-window">
            <div class="chat-header">
                <h3>PK Travels Assistant</h3>
                <button style="background: none; border: none; cursor: pointer;" onclick="toggleChat()">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">
                        <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
                    </svg>
                </button>
            </div>
            
            <div class="chat-messages">
        """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f'<div class="message-container"><div class="user-message">{message["content"]}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-container"><div class="bot-message">{message["content"]}</div></div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input area
        st.markdown('<div class="chat-input">', unsafe_allow_html=True)
        
        # Use a form to properly handle input submission and clearing
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask about traveling in Pakistan...", key="chat_input_field")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                send_message(user_input)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add a footer
st.markdown("""
<div style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #e0e0e0;">
    <p>© 2023 PK Travels. All rights reserved.</p>
    <p>Contact us: info@pktravels.com | Phone: +92-51-1234567</p>
    <p>Address: Blue Area, Islamabad, Pakistan</p>
</div>
""", unsafe_allow_html=True)

# Save chat history when the session ends
def save_chat_history():
    if st.session_state.chat_history:
        # Ensure the chat_logs directory exists
        os.makedirs("chat_logs", exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"chat_logs/session_{timestamp}.txt"
        
        # Save the chat history
        with open(filename, "w", encoding="utf-8") as f:
            for message in st.session_state.chat_history:
                role = "User" if message["role"] == "user" else "PK Travels AI"
                f.write(f"{role}: {message['content']}\n")

# Register the function to run when Streamlit script terminates
import atexit
atexit.register(save_chat_history) 