from dotenv import load_dotenv
load_dotenv()  # Take environment variables from .env.

import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import numpy as np
import cv2
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Initialize Streamlit app configuration in wide mode
st.set_page_config(page_title="ChatFusion", layout="wide")

# Initialize Google API key
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state to store chat history and uploaded image
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

if 'tab' not in st.session_state:
    st.session_state['tab'] = 'New Chat'

if 'conversation_ids' not in st.session_state:
    st.session_state['conversation_ids'] = []

if 'current_conversation_id' not in st.session_state:
    st.session_state['current_conversation_id'] = None

if 'current_message_index' not in st.session_state:
    st.session_state['current_message_index'] = -1  # Default to -1 to start a new conversation

# Function to load the new model and get responses
def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
    if input != "":
        if image is not None:
            response = model.generate_content([input, image])
        else:
            response = model.generate_content([input])
    else:
        response = model.generate_content([image])
    return response.text

# Function to combine chat history for context
def get_combined_input():
    history = st.session_state['history']
    # Combine the last few interactions (adjust length as needed)
    conversation_history = "\n".join([f"User: {chat['input']}\n\nBot: {chat['response']}" for chat in history[-3:]])
    # Add the current input to the history
    return f"{conversation_history}\nUser: {st.session_state.input}"

# Function to handle input submission
def handle_submit():
    try:
        # Combine history with current input
        combined_input = get_combined_input()
        
        # Get the image from session state
        uploaded_image = st.session_state['uploaded_image']
        
        # Check if there is either an input or an image
        if combined_input.strip() or uploaded_image:
            response = get_gemini_response(combined_input, uploaded_image)
            
            # Save the input, response, and image to session state history
            st.session_state['history'].append({
                'input': st.session_state.input,
                'response': response,
                'image': uploaded_image  # Save the uploaded image
            })
            
            # Clear the input and uploaded image after submission
            st.session_state.input = ""
            st.session_state.uploaded_image = None
            
            # Update the displayed response
            st.session_state['response'] = response
        else:
            st.error("Please provide either an input prompt or an image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Function to start a new chat
def start_new_chat():
    if st.session_state['history']:
        if st.session_state['current_conversation_id'] is None:
            # Create a new conversation ID if one doesn't exist
            conversation_id = len(st.session_state['conversation_ids']) + 1
        else:
            # Reuse the current conversation ID
            conversation_id = st.session_state['current_conversation_id']
        
        # Save the current chat history
        st.session_state[f'conversation_{conversation_id}'] = st.session_state['history'][:]
        if conversation_id not in st.session_state['conversation_ids']:
            st.session_state['conversation_ids'].append(conversation_id)
        
        # Clear history for new chat
        st.session_state['history'] = []
        st.session_state['current_conversation_id'] = None
        st.session_state['current_message_index'] = -1  # Reset message index for new chat

# Function to generate PDF
from textwrap import wrap


from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import io

# Register the necessary fonts
pdfmetrics.registerFont(TTFont('NotoSansGurmukhi', 'NotoSansGurmukhi.ttf'))
pdfmetrics.registerFont(TTFont('NotoSansDevanagari', 'NotoSansDevanagari.ttf'))




from textwrap import wrap
def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    margin = 10  # Margin for content
    y = height - margin  # Start at the top of the page for content

    for idx, chat in enumerate(st.session_state['history']):
        # Get the user input and chatbot response
        prompt_text = f"User: {chat['input']}"
        response_text = f"Bot: {chat['response']}"

        # Function to select font based on the detected language
        def select_font(text):
            if any('\u0A00' <= char <= '\u0A7F' for char in text):  # Gurmukhi (Punjabi)
                c.setFont("NotoSansGurmukhi", 12)
            elif any('\u0900' <= char <= '\u097F' for char in text):  # Devanagari (Hindi)
                c.setFont("NotoSansDevanagari", 12)
            else:
                c.setFont("Helvetica", 12)  # Default font for English

        # Set text color to black
        c.setFillColorRGB(0, 0, 0)

        # Draw the prompt with appropriate font
        select_font(prompt_text)
        for line in wrap(prompt_text, width=90):
            c.drawString(margin, y, line)
            y -= 12  # Line height

        # Add extra space after the prompt
        y -= 12

        # Draw the response with appropriate font
        select_font(response_text)
        for line in wrap(response_text, width=90):
            c.drawString(margin, y, line)
            y -= 12  # Line height

        # Add extra space after the response
        y -= 12

        # Print the image if available
        if chat.get('image'):
            image_path = f"temp_image_{idx + 1}.png"
            chat['image'].save(image_path)
            image_width, image_height = 100, 100
            if y - image_height < margin:
                c.showPage()
                y = height - margin - image_height

            c.drawImage(image_path, margin, y - image_height, width=image_width, height=image_height)
            y -= (image_height + 20)

        # Check if we need a new page
        if y < margin:
            c.showPage()
            y = height - margin

        # Add watermark at the top-right corner, very close to the border
        c.setFont("Helvetica", 10)  # Smaller font for the watermark
        c.setFillColorRGB(0.7, 0.7, 0.7)  # Light grey color for watermark to keep it readable but subtle
        c.drawRightString(width - 10, height - 15, "Generated by ChatFusion")

    c.save()
    buffer.seek(0)
    return buffer


# Title at the top left
st.markdown("<h1 style='text-align: left;'>ChatFusion</h1>", unsafe_allow_html=True)

# Create layout for the app
col1, col2 = st.columns([2, 1])  # Adjust column ratio as needed

# Sidebar for controls and history
with st.sidebar:
 

    

    # Start New Chat button with icon
    if st.button('💬 Start New Chat'):
        start_new_chat()
        st.session_state['tab'] = 'New Chat'

    # Display chat history tabs with icons
    st.header("▶ Saved Conversations")

# Check if there are any saved conversations
    if 'conversation_ids' in st.session_state and st.session_state['conversation_ids']:
    # Loop through the saved conversation IDs
      for conversation_id in st.session_state['conversation_ids']:
        if st.button(f"🔸 Chat {conversation_id}"):
            st.session_state['tab'] = 'Chat History'
            st.session_state['history'] = st.session_state[f'conversation_{conversation_id}']
            st.session_state['current_conversation_id'] = conversation_id
            st.session_state['current_message_index'] = len(st.session_state['history']) - 1
    else:
      st.write("No chats found.")

    # Clear history button with icon
    
    st.sidebar.markdown("---") 
    if st.button('🗑️ Clear History'):
        st.session_state['conversation_ids'] = []
        st.session_state['history'] = []
        st.write("Chat history cleared.")

    # Generate PDF button with icon
    if st.button('📄 Generate PDF'):
        pdf_buffer = generate_pdf()
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="conversation_history.pdf",
            mime="application/pdf"
        )

# Main column for chat response or history
with col1:
    if st.session_state['tab'] == 'New Chat':
        # Display full chat history
        if st.session_state['history']:
            st.subheader("Chat History")
            for idx, chat in enumerate(st.session_state['history']):
                st.write(f"User: {chat['input']}")
                st.write(f"🤖: {chat['response']}")
                # Display the stored image in the history if it exists
                if chat['image']:
                    st.image(chat['image'], caption=f"Image {idx + 1}", use_column_width=True)
        else:
            st.subheader("Chat with me!")
            st.write("Start a new chat by entering a prompt or uploading an image.")

        # Input panel at the bottom of the main column
        input_container = st.container()
        with input_container:
            st.text_input("Input Prompt: ", key="input", on_change=handle_submit)
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.session_state['uploaded_image'] = image
                st.image(image, caption="Uploaded Image.", use_column_width=True)
            
        

    elif st.session_state['tab'] == 'Chat History':
        st.subheader("Chat History")
        if st.session_state['history']:
            for idx, chat in enumerate(st.session_state['history']):
                st.write(f"*Prompt {idx + 1}:* {chat['input']}")
                st.write(f"*Response {idx + 1}:* {chat['response']}")
                if chat['image']:
                    st.image(chat['image'], caption=f"Image {idx + 1}", use_column_width=True)
        else:
            st.write("No chat history available.")

# Sidebar section for "Start New Chat" and "Audio to Text Converter"

import streamlit as st
from pydub import AudioSegment, silence
import speech_recognition as sr

# Initialize recognizer
recognizer = sr.Recognizer()

# Sidebar configuration
st.sidebar.markdown("<h3 style='font-size:20px;'>▶ More Features</h3>", unsafe_allow_html=True)

# Dropdown button for Audio to Text Converter in the sidebar
pdf_summarizer_option = st.sidebar.selectbox(
    "Choose Action",  # Label for dropdown
    ("No option selected", "Audio to Text Converter","PDF Bot","Learn to Pronounce")  # Blank option as default, Audio to Text Converter as only other option
)

# If 'Audio to Text Converter' is selected, show the audio uploader and conversion section in the sidebar
if pdf_summarizer_option == "Audio to Text Converter":
    # Heading for the Audio to Text converter (within sidebar)
    st.sidebar.markdown("---") 
    st.sidebar.markdown("<h3 style='text-align: left;'>Audio to Text Converter</h3>", unsafe_allow_html=True)

    # Upload audio file in the sidebar
    audio = st.sidebar.file_uploader("Upload Your Audio File (mp3, wav)", type=["mp3", "wav"])

    # Button to convert audio to text in the sidebar
    convert_button = st.sidebar.button("Convert Audio to Text")

    # Clear button functionality in the sidebar
    clear_button = st.sidebar.button("Clear")
   

    # Handle audio-to-text conversion if the button is pressed
    if audio and convert_button:
        import time


# Create a placeholder in the Streamlit app
        placeholder = st.empty()

# Display the temporary message
        with placeholder:
            st.write("Processing audio...")
# Simulate audio processing with a sleep function
            time.sleep(5)  # Replace this with your actual audio processing code

# Once processing is complete, clear the message and show the output
            placeholder.empty()
            st.write("Audio processing complete!")
# Display the actual output here

        # Load the audio file with pydub
        audio_segment = AudioSegment.from_file(audio)

        # Split the audio file into chunks based on silence
        chunks = silence.split_on_silence(
            audio_segment,
            min_silence_len=500,  # Minimum length of silence to split on (in ms)
            silence_thresh=audio_segment.dBFS - 20,  # Silence threshold
            keep_silence=100  # Keep 100ms of silence at the beginning/end of each chunk
        )

        text_output = ""

        for index, chunk in enumerate(chunks):
            chunk.export(f"chunk_{index}.wav", format="wav")  # Save chunk as wav file

            # Recognize audio from the chunk
            with sr.AudioFile(f"chunk_{index}.wav") as source:
                recorded = recognizer.record(source)

            try:
                text = recognizer.recognize_google(recorded)
                text_output += f"🤖: {text}\n"
            except sr.UnknownValueError:
                text_output += f"🤖: Could not understand audio.\n"
            except sr.RequestError as e:
                text_output += f"🤖: Could not request results from Google Speech Recognition service; {e}\n"

        # Display the recognized text after processing all chunks on the main screen
        st.markdown("<h4 style='text-align: left;'>Recognized Text:</h4>", unsafe_allow_html=True)
        st.text_area("Output", text_output, height=200)

    # Clear button functionality
    if clear_button:
        st.write("State cleared!")
        

# If no option is selected (empty default), the sidebar remains blank
else:
    st.sidebar.write("")


import streamlit as st
import os
import requests
from pypdf import PdfReader

# Folder to save uploaded PDF files
upload_folder = 'uploaded_pdf_file'

# Ensure the upload folder exists
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

# Initialize session state keys
if "messages" not in st.session_state:
    st.session_state.messages = []
if "cleared_message" not in st.session_state:
    st.session_state.cleared_message = ""
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""

def clear_state():
    st.session_state.messages = []
    st.session_state.cleared_message = "State cleared!"
    st.session_state.raw_text = ""

# If "PDF Bot" option is selected
if pdf_summarizer_option == "PDF Bot":
    st.sidebar.markdown("---") 
    st.sidebar.markdown("<h3 style='text-align: left;'>Chat with PDF Bot</h3>", unsafe_allow_html=True)

    # PDF file uploader widget in sidebar
    pdf_docs = st.sidebar.file_uploader("Choose PDF Files", accept_multiple_files=True, type=["pdf"])

    # Button to extract text from uploaded PDFs
    extract_button = st.sidebar.button("Extract Text")

    # Sidebar text input for user question
    user_question = st.sidebar.text_input("Ask a question based on the PDF content:", key="user_question")
    submit_button = st.sidebar.button("Submit & Process")

    if st.sidebar.button("Clear"):
        clear_state()

    # Display cleared message if it exists in session state
    if "cleared_message" in st.session_state and st.session_state.cleared_message:
        st.success(st.session_state.cleared_message)
        st.session_state.cleared_message = ""

    # Hugging Face API function
    def response_generator(prompt, context):
        API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
        headers = {"Authorization": "Bearer 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"}

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        payload = {
            "inputs": {
                "question": prompt,
                "context": context
            }
        }

        output = query(payload)
        # Ensure output is valid
        return output.get('answer', 'No answer found.') if isinstance(output, dict) else "API Error: Unable to fetch answer."

    # If Extract Text button is clicked
    if extract_button:
        if pdf_docs:
            raw_text = ""
            for pdf in pdf_docs:
                # Get the file name and the save path
                file_name = pdf.name
                saved_path = os.path.join(upload_folder, file_name)

                # Save the uploaded PDF file locally
                with open(saved_path, 'wb') as f:
                    f.write(pdf.getbuffer())
                st.success(f'PDF File has been successfully uploaded to {saved_path}')

                # Read and extract text from the first page of the PDF
                reader = PdfReader(saved_path)
                if len(reader.pages) > 0:
                    first_page = reader.pages[0]
                    raw_text += first_page.extract_text()

            # Store extracted text in session state
            if raw_text:
                st.session_state.raw_text = raw_text  # Save to session state
            else:
                st.warning("No text extracted from the first page of the PDFs.")
        else:
            st.warning("No PDF files uploaded. Please upload at least one PDF file.")

    # Display extracted text from the first page if available
    if "raw_text" in st.session_state and st.session_state.raw_text:
        st.header("Extracted Text from PDF's First Page")
        st.write(st.session_state.raw_text)  # Display full text from the first page

    # If Submit & Process button is clicked
    if submit_button:
        if "raw_text" in st.session_state and st.session_state.raw_text:  # Check if raw_text is available
            if user_question:  # Check if the user has asked a question
                with st.spinner("Getting the answer..."):
                    response = response_generator(user_question, st.session_state.raw_text)
                    answer = response

                # Display the user question and the assistant's response
                st.header("Output")

                st.write(f"**User:** {user_question}")
                st.write(f"**🤖:** {answer}")

                # Save chat history in session state
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                st.session_state.messages.append({"role": "user", "content": user_question})
                st.session_state.messages.append({"role": "Bot", "content": answer})
            else:
                st.warning("Please enter a question before submitting.")
        else:
            st.warning("No text extracted from the PDFs. Please extract text first.")

# Display chat history with a header
if "messages" in st.session_state and st.session_state.messages:
    st.subheader("Recents (up to bottom)")
    for message in st.session_state.messages:
        role = "You" if message["role"] == "user" else "Bot"
        st.write(f"**{role}:** {message['content']}")


import streamlit as st
from gtts import gTTS
import os
import pygame

# Function to convert text to voice
def text_to_voice(text):
    if text:
        st.write(f"Converting Text to Voice: {text}")
        tts = gTTS(text=text, lang='en')
        file_path = os.path.join(os.getcwd(), "output.mp3")
        tts.save(file_path)
        st.write(f"File saved at: {file_path}")

        # Initialize pygame mixer
        pygame.mixer.init()
    
        # Load and play the audio file
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()

        # Wait for the audio to finish playing
        while pygame.mixer.music.get_busy():
            continue

        # Remove the audio file after playback
        os.remove(file_path)
    else:
        st.write("No text to convert to speech.")

# Main function to handle the Streamlit app
def main():
    st.title("Text to Audio Converter")
    
    text = st.text_input("Enter text to convert to voice:")
    
    if st.button("Convert to Voice"):
        text_to_voice(text)

# Run the app
if __name__ == "__main__":
    main()




footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 100%;
        text-align: left;
        color: #FFA500;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        Built by Prabhdeep Singh
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)




import streamlit as st
import base64
import pickle
import os

# Define a function to save the checkbox states to a file
def save_state():
    with open('background_state.pkl', 'wb') as f:
        pickle.dump({
            'show_dynamic_bg': st.session_state.show_dynamic_bg,
            'show_dark_wallpaper': st.session_state.show_dark_wallpaper
        }, f)

# Define a function to load the checkbox states from a file
def load_state():
    if os.path.exists('background_state.pkl'):
        with open('background_state.pkl', 'rb') as f:
            state = pickle.load(f)
            st.session_state.show_dynamic_bg = state.get('show_dynamic_bg', True)  # Default to True for permanent check
            st.session_state.show_dark_wallpaper = state.get('show_dark_wallpaper', True)  # Default to True for permanent check

# Load previous session state
load_state()

# Initialize session state for background options if not already done
if 'show_dynamic_bg' not in st.session_state:
    st.session_state.show_dynamic_bg = True  # Default to True for permanent check
if 'show_dark_wallpaper' not in st.session_state:
    st.session_state.show_dark_wallpaper = True  # Default to True for permanent check

# Function to add dynamic background using CSS
def add_dynamic_background():
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(270deg, #1e3c72, #2a5298);
            background-size: 400% 400%;
            animation: gradientBackground 15s ease infinite;
        }

        @keyframes gradientBackground {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        .stApp {
            background-color: rgba(0, 0, 0, 0);
            background-image: url('https://www.transparenttextures.com/patterns/asfalt-dark.png');
            background-blend-mode: overlay;
            animation: moveBackground 15s infinite linear;
        }

        @keyframes moveBackground {
            0% {background-position: 0% 0%;}
            100% {background-position: 100% 100%;}
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to add dark wallpaper
def add_dark_wallpaper(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Sidebar for background options
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='text-align: left;'>Fusion Background</h3>", unsafe_allow_html=True)

# Checkbox for dynamic background
st.session_state.show_dynamic_bg = st.sidebar.checkbox("Show Dynamic Background/White", value=st.session_state.show_dynamic_bg)

# Checkbox for dark wallpaper
st.session_state.show_dark_wallpaper = st.sidebar.checkbox("Show Dark Wallpaper", value=st.session_state.show_dark_wallpaper)
st.sidebar.markdown("---")

# Save the state when the checkboxes are changed
save_state()

# If the dynamic background checkbox is checked, show the dynamic background
if st.session_state.show_dynamic_bg:
    add_dynamic_background()
   

# If the dark wallpaper checkbox is checked, show the dark wallpaper
if st.session_state.show_dark_wallpaper:
    add_dark_wallpaper("chatfusiondark.png")
   




# Custom CSS to style the button and transparent background
import streamlit as st

st.markdown(
    """
    <style>
    .transparent-button {
        background-color: rgba(255, 255, 255, 0.5);  /* Transparent background */
        border: none;  /* Remove border */
        padding: 10px 20px;  /* Padding for the button */
        border-radius: 5px;  /* Rounded corners */
        cursor: pointer;  /* Pointer cursor on hover */
        font-size: 16px;  /* Font size */
        color: black;  /* Text color */
        position: fixed;
        top: 20px;  /* Position the button 20px from the top */
        right: 20px;  /* Position the button 20px from the right */
        z-index: 9999;  /* Ensure it stays on top of other elements */
    }
    .about-section {
        padding: 20px;
        border: 1px solid #e0e0e0;  /* Optional: border for separation */
        border-radius: 5px;  /* Optional: rounded corners */
        margin-top: 40px;  /* Add margin to push the section below the button */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Create the About button in the sidebar with custom styling
if st.sidebar.button("About", key="about", help="Click to learn more about this app"):
    # Toggle the show_about state
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False
    
    st.session_state.show_about = not st.session_state.show_about

# Show or hide the About section in the main content based on the session state
if 'show_about' in st.session_state and st.session_state.show_about:
    st.markdown("""
        <div class="about-section">
        <h2>About this App</h2>
        <p>This application is built by Prabhdeep Singh.</p>
        
        ### Features
        It integrates several features to enhance user interaction:
        - **PDF Chatbot**: Interact with PDF documents seamlessly.
        - **Audio to Text Converter**: Convert audio files into text easily.
        - **Image Recognition**: Upload images or use the webcam for analysis.
        - **Natural Language Processing**: Facilitates better communication.
        - **User-friendly Interface**: Simple and easy to navigate.
        - **Visually Appealing Fusion Backgrounds**: Add a cool and modern look.
        
        This app combines image recognition and natural language processing (NLP) for effective user interaction.
        </div>
    """, unsafe_allow_html=True)
    







