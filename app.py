import streamlit as st
import torch
from email_agent import EmailAgent
import time
import asyncio

# Set page config
st.set_page_config(
    page_title="Email Agent",
    page_icon="✉️",
    layout="wide"
)

# Initialize session state
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = EmailAgent()
    except Exception as e:
        st.error(f"Error initializing email agent: {str(e)}")
        st.stop()

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        height: 200px;
    }
    .result-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: #000000;
    }
    .result-box h4 {
        color: #000000;
        margin-bottom: 10px;
    }
    .sentiment-positive {
        color: #00aa00;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #ff0000;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# App title and description
st.title("✉️ Smart Email Agent")
st.markdown("""
    This agent can analyze your emails, summarize them, detect sentiment, and generate appropriate responses.
    Simply paste your email below and let the agent do its magic!
""")

# Email input
email_text = st.text_area(
    "Enter your email here:",
    placeholder="Paste your email content here...",
    height=400,
    max_chars=5000
)

# Add character counter
if email_text:
    st.caption(f"Characters: {len(email_text)}/5000")

# Process button
if st.button("Analyze Email", type="primary"):
    if email_text.strip():
        try:
            with st.spinner("Processing email..."):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Step 1: Analyze email
                progress_bar.progress(25)
                summary, sentiment, sentiment_score = st.session_state.agent.analyze_email(email_text)
                
                # Step 2: Determine if response is needed and its type
                progress_bar.progress(50)
                needs_reply, response_type = st.session_state.agent.needs_response(sentiment, sentiment_score)
                
                # Step 3: Generate response if needed
                progress_bar.progress(75)
                if needs_reply:
                    response = st.session_state.agent.generate_response(email_text, summary, response_type)
                
                progress_bar.progress(100)
                time.sleep(0.5)  # Small delay for visual effect
                
            # Display results
            st.markdown("### 📝 Analysis Results")
            
            # Summary
            st.markdown("#### Summary")
            st.markdown(f'<div class="result-box"><p style="color: #000000;">{summary}</p></div>', unsafe_allow_html=True)
            
            # Sentiment
            st.markdown("#### Sentiment Analysis")
            sentiment_class = "sentiment-positive" if sentiment == "POSITIVE" else "sentiment-negative"
            st.markdown(f'<div class="result-box">'
                       f'<p style="color: #000000;">Sentiment: <span class="{sentiment_class}">{sentiment}</span> '
                       f'(Score: {sentiment_score:.2f})</p>'
                       f'</div>', unsafe_allow_html=True)
            
            # Response needed
            st.markdown("#### Response Decision")
            response_decision = "Yes" if needs_reply else "No"
            response_type_display = f" ({response_type})" if needs_reply else ""
            st.markdown(f'<div class="result-box"><p style="color: #000000;">Response needed: {response_decision}{response_type_display}</p></div>', unsafe_allow_html=True)
            
            # Generated response
            if needs_reply:
                st.markdown("#### Generated Response")
                st.markdown(f'<div class="result-box"><p style="color: #000000;">{response}</p></div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while processing the email: {str(e)}")
    else:
        st.warning("Please enter an email to analyze.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit") 