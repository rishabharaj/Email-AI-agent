import nltk
from transformers import pipeline
import torch
from typing import Tuple
import os
from dotenv import load_dotenv

# Download required NLTK data
try:
    nltk.download('punkt')
except Exception as e:
    print(f"Warning: Could not download NLTK data: {str(e)}")

class EmailAgent:
    def __init__(self):
        try:
            # Initialize sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device="cpu"
            )
            
            # Initialize summarization pipeline with more concise parameters
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device="cpu"
            )
            
            # Initialize text generation pipeline for responses
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                device="cpu",
                model_kwargs={"max_length": 1024}  # Set model's max length
            )
        except Exception as e:
            raise Exception(f"Failed to initialize pipelines: {str(e)}")

    def analyze_email(self, email_text: str) -> Tuple[str, str, float]:
        """
        Analyze the email and return summary, sentiment, and sentiment score.
        """
        try:
            # Generate more concise summary
            summary = self.summarizer(
                email_text,
                max_length=150,
                min_length=30,
                do_sample=False,
                num_beams=4,
                length_penalty=0.8
            )[0]['summary_text']
            
            # Analyze sentiment
            sentiment_result = self.sentiment_analyzer(email_text)[0]
            sentiment = sentiment_result['label']
            sentiment_score = sentiment_result['score']
            
            return summary, sentiment, sentiment_score
        except Exception as e:
            raise Exception(f"Failed to analyze email: {str(e)}")

    def needs_response(self, sentiment: str, sentiment_score: float) -> Tuple[bool, str]:
        """
        Determine if the email needs a response based on sentiment analysis.
        Returns a tuple of (needs_response, response_type)
        """
        try:
            if sentiment == "NEGATIVE":
                return True, "detailed"  # Negative emails need detailed responses
            elif sentiment == "POSITIVE" and sentiment_score > 0.9:
                return True, "brief"  # Very positive emails get brief responses
            return False, "none"
        except Exception as e:
            raise Exception(f"Failed to determine response need: {str(e)}")

    def generate_response(self, email_text: str, summary: str, response_type: str) -> str:
        """
        Generate a response based on the email content and summary.
        """
        try:
            if response_type == "detailed":
                # Generate a detailed response for negative emails
                prompt = f"""Generate a positive and professional email response to this email summary: '{summary}'. 
                Format the response as a proper email with:
                1. A warm greeting
                2. Positive acknowledgment of the concerns
                3. Constructive solutions and next steps
                4. A positive closing
                Keep the tone optimistic, professional, and solution-oriented.
                Focus on positive outcomes and collaboration.
                Response:"""
                max_new_tokens = 200
            else:
                # Generate a brief response for positive emails
                prompt = f"""Generate a warm and positive email response to this email summary: '{summary}'. 
                Format as a proper email with greeting and closing.
                Keep it to 2-3 sentences.
                Make it appreciative and encouraging.
                Response:"""
                max_new_tokens = 100

            response = self.text_generator(
                prompt,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=50256,
                truncation=True,
                return_full_text=False  # Don't include the prompt in the output
            )[0]['generated_text']
            
            # Extract just the response part and clean it up
            response = response.strip()
            
            # Ensure proper email format with positive tone
            if not response.startswith("Dear") and not response.startswith("Hello"):
                response = "Dear Team,\n\n" + response
            
            if not response.endswith("Best regards,") and not response.endswith("Regards,"):
                response = response + "\n\nBest regards,"
            
            # Remove any incomplete sentences at the end
            if '.' in response:
                last_period = response.rindex('.')
                if last_period < len(response) - 1:
                    response = response[:last_period + 1]
            
            # Ensure positive tone in the response
            positive_phrases = [
                "Thank you for bringing this to our attention",
                "We appreciate your feedback",
                "We're committed to finding a solution",
                "We look forward to working together",
                "We're excited to address this",
                "We value your input",
                "We're confident we can resolve this",
                "We're happy to help",
                "We're glad to assist",
                "We're here to support you"
            ]
            
            # Add a positive phrase if the response seems too neutral
            if not any(phrase.lower() in response.lower() for phrase in positive_phrases):
                response = response.replace("Dear Team,\n\n", f"Dear Team,\n\n{positive_phrases[0]}. ")
            
            return response
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")

    def process_email(self, email_text: str):
        """
        Process the email through all steps and print results.
        """
        print("\nProcessing email...")
        
        try:
            # Step 1: Analyze email
            summary, sentiment, sentiment_score = self.analyze_email(email_text)
            
            print("\n=== Analysis Results ===")
            print(f"Summary: {summary}")
            print(f"Sentiment: {sentiment} (Score: {sentiment_score:.2f})")
            
            # Step 2: Determine if response is needed and its type
            needs_reply, response_type = self.needs_response(sentiment, sentiment_score)
            print(f"\nResponse needed: {'Yes' if needs_reply else 'No'}")
            if needs_reply:
                print(f"Response type: {response_type}")
            
            # Step 3: Generate and print response if needed
            if needs_reply:
                response = self.generate_response(email_text, summary, response_type)
                print("\n=== Generated Response ===")
                print(response)
        except Exception as e:
            print(f"Error processing email: {str(e)}")

def main():
    # Example email for testing
    example_email = """
    Dear Team,
    
    I hope this email finds you well. I wanted to bring to your attention some concerns I have regarding the recent project timeline. 
    The current schedule seems unrealistic given the scope of work, and I'm worried we might not meet the deadline. 
    Additionally, there have been some communication issues between departments that are causing delays.
    
    I would appreciate it if we could schedule a meeting to discuss these matters and find a way forward.
    
    Best regards,
    John
    """
    
    try:
        agent = EmailAgent()
        agent.process_email(example_email)
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 