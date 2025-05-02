# Email AI Agent 🤖

A Python-based email analysis and response generation system that uses NLP models to analyze emails, detect sentiment, generate summaries, and create appropriate responses.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.22.0-orange)](https://streamlit.io)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/rishabharaj/Email-AI-agent)

## Features ✨

- 📧 Email Analysis
- 😊 Sentiment Detection
- 📝 Smart Summarization
- 💬 Response Generation
- 🎨 Modern Web Interface
- ⚡ Real-time Processing

## Technical Architecture 🏗️

The project consists of two main components:

1. **Core Engine** (`email_agent.py`):
   - Email analysis and processing
   - Sentiment detection using DistilBERT
   - Summarization using BART
   - Response generation using GPT-2

2. **Web Interface** (`app.py`):
   - Streamlit-based UI
   - Real-time processing
   - Beautiful and responsive design

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/rishabharaj/Email-AI-agent.git
cd Email-AI-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 📖

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the provided URL (usually http://localhost:8501)

3. Paste your email text and click "Analyze Email"

## Project Structure 📁

```
Email-AI-agent/
├── app.py              # Streamlit web interface
├── email_agent.py      # Core email processing logic
├── requirements.txt    # Project dependencies
├── README.md          # Project documentation
└── .gitignore         # Git ignore file
```

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- [Hugging Face](https://huggingface.co/) for the transformer models
- [Streamlit](https://streamlit.io/) for the web framework
- [NLTK](https://www.nltk.org/) for natural language processing tools

## 🚀 Features

- **Email Analysis**: Automatically analyzes email content
- **Sentiment Detection**: Identifies the emotional tone of the email
- **Smart Summarization**: Generates concise summaries of email content
- **Response Generation**: Creates appropriate responses when needed
- **Modern Web Interface**: Beautiful and user-friendly UI using Streamlit
- **Real-time Processing**: Instant analysis and results

## 🛠️ Technical Architecture

### Core Components

1. **EmailAgent Class** (`email_agent.py`)
   - Uses three main NLP pipelines:
     - **Sentiment Analysis**: Using DistilBERT model
     - **Summarization**: Using BART model
     - **Text Generation**: Using GPT-2 model
   - Key Functions:
     - `analyze_email()`: Performs simultaneous summarization and sentiment analysis
     - `needs_response()`: Determines if a response is needed based on sentiment
     - `generate_response()`: Creates a response using the email summary
     - `process_email()`: Orchestrates the entire process

2. **Web Interface** (`app.py`)
   - Built with Streamlit
   - Features:
     - Clean, modern UI design
     - Real-time progress tracking
     - Color-coded sentiment analysis
     - Responsive layout
     - Error handling and user feedback

### NLP Models Used

1. **DistilBERT** (for sentiment analysis)
   - Model: `distilbert-base-uncased-finetuned-sst-2-english`
   - Purpose: Analyzes email sentiment (positive/negative)
   - Output: Sentiment label and confidence score

2. **BART** (for summarization)
   - Model: `facebook/bart-large-cnn`
   - Purpose: Generates concise email summaries
   - Parameters:
     - max_length: 80 characters
     - min_length: 20 characters
     - num_beams: 4
     - length_penalty: 0.8

3. **GPT-2** (for response generation)
   - Model: `gpt2`
   - Purpose: Generates appropriate email responses
   - Parameters:
     - max_length: 100 characters
     - temperature: 0.7
     - top_p: 0.9
     - repetition_penalty: 1.2

## 🧰 Dependencies

- `torch`: PyTorch for deep learning operations
- `transformers`: Hugging Face Transformers library for NLP models
- `nltk`: Natural Language Toolkit for text processing
- `streamlit`: Web application framework
- `python-dotenv`: Environment variable management

## 🚀 Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
.\venv\Scripts\activate
```
- Linux/Mac:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📝 Usage Guide

1. **Input Email**
   - Paste your email text into the input area
   - Click "Analyze Email" button

2. **Analysis Process**
   - The system will:
     1. Generate a concise summary
     2. Analyze sentiment
     3. Determine if a response is needed
     4. Generate a response (if required)

3. **Results Display**
   - Summary of the email
   - Sentiment analysis with color coding
   - Response decision
   - Generated response (if applicable)

## 🔧 Customization

You can customize the following aspects:

1. **Summary Length**
   - Adjust `max_length` and `min_length` in `analyze_email()`
   - Modify `length_penalty` for different summary styles

2. **Response Generation**
   - Change prompt in `generate_response()`
   - Adjust generation parameters (temperature, top_p, etc.)

3. **UI Styling**
   - Modify CSS in `app.py`
   - Adjust layout and colors

## 📚 Technical Details

### How It Works

1. **Email Analysis**
   - Text is processed through BART model for summarization
   - DistilBERT analyzes sentiment simultaneously
   - Results are combined for comprehensive analysis

2. **Response Decision**
   - Based on sentiment and confidence score
   - Triggers response generation when:
     - Sentiment is negative, or
     - Positive sentiment with high confidence (>0.9)

3. **Response Generation**
   - Uses email summary as context
   - Generates professional, concise responses
   - Ensures complete sentences and proper formatting

## ⚠️ Note

- First run might take time as it downloads ML models
- Requires stable internet connection for model downloads
- Processing time depends on email length and complexity

## 🤝 Contributing

Feel free to contribute to this project by:
1. Forking the repository
2. Creating a feature branch
3. Submitting a pull request

## 📄 License

This project is open-source and available under the MIT License. 