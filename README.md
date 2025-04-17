# PK Travels Chatbot Application

A travel service chatbot for PK Travels that provides information about travel destinations, packages, and services using AI assistance.

## Features

- Interactive chatbot interface in the bottom left corner of the website
- Knowledge-base powered travel assistance
- Web scraping capability to gather travel information
- Text-to-speech capability for accessibility
- Chat history logging
- Responsive Streamlit web interface

## Setup Instructions

### Prerequisites

- Python 3.8+
- Pip (Python package installer)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd pk-travels-chatbot
   ```

2. Create and activate a virtual environment (recommended):
   ```
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key
   ```

### Running the Application

1. Create a knowledge base by adding text or PDF files to the `KB` folder. These will be used by the chatbot to answer questions about PK Travels.

2. Run the Streamlit application:
   ```
   streamlit run streamlit_app.py
   ```

3. The application will open in your default web browser at `http://localhost:8501`.

## Adding Knowledge Base Content

There are two ways to add content to the knowledge base:

1. **Manual addition**: Add text (.txt) or PDF (.pdf) files directly to the `KB` folder.

2. **Web scraping**: When you run the application for the first time with an empty KB folder, it will ask if you want to scrape data from websites. You can then provide URLs of travel-related websites to build the knowledge base.

## Chat Interface

- Click the chat button in the bottom left corner to open the chat interface
- Type your travel-related question in the input field
- The AI will respond based on the knowledge base information
- Chat history is saved in the `chat_logs` folder

## Customization

- Replace the placeholder images with actual PK Travels logo and destination images
- Update the featured destinations with actual travel packages
- Modify the system prompt in the `app_safe.py` file to better reflect PK Travels' services and brand voice

## Files Description

- `app_safe.py`: Main chatbot backend implementation
- `streamlit_app.py`: Streamlit web interface
- `requirements.txt`: Required Python packages
- `KB/`: Knowledge base folder for storing travel information
- `chat_logs/`: Folder for storing chat conversation logs

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with Streamlit, LangChain, and GROQ API
- Uses HuggingFace models for embeddings
- Implements ChromaDB for vector storage 