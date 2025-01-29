from flask import Flask, request, jsonify
from pathlib import Path
import google.generativeai as genai
import os
import time
import logging
from werkzeug.utils import secure_filename
import tempfile
from dotenv import load_dotenv
from flask_cors import CORS  # Import CORS

# Load environment variables
load_dotenv()


# Configure the GenerativeAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up the model configuration for text generation
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

# Define safety settings for content generation
safety_settings = [
    {"category": f"HARM_CATEGORY_{category}", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
    for category in ["HARASSMENT", "HATE_SPEECH", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
]

# Initialize the GenerativeModel
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings,
)

# Create Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)  # This will allow all origins

# Function to read image data from a file path
def read_image_data(file_path):
    image_path = Path(file_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Could not find image: {image_path}")
    return {"mime_type": "image/jpeg", "data": image_path.read_bytes()}

# Retry logic with exponential backoff
def retry_request(func, retries=3, backoff_in_seconds=1):
    for i in range(retries):
        try:
            return func()
        except Exception as e:
            logging.warning(f"Attempt {i+1} failed: {e}")
            if i < retries - 1:
                time.sleep(backoff_in_seconds * (2 ** i))
            else:
                raise e

# Function to generate a response based on a prompt and an image path
def generate_gemini_response(prompt, image_path):
    image_data = read_image_data(image_path)

    # Define the function to be retried
    def api_call():
        return model.generate_content([prompt, image_data])

    # Make the API call with retry logic
    response = retry_request(api_call)
    return response.text

# Initial input prompt for the plant pathologist
input_prompt = """
As a highly skilled plant pathologist, your expertise is indispensable in our pursuit of maintaining optimal plant health. You will be provided with information or samples related to plant diseases, and your role involves conducting a detailed analysis to identify the specific issues, propose solutions, and offer recommendations.

**Analysis Guidelines:**

1. **Disease_Identification:** Examine the provided information or samples to identify and characterize plant diseases accurately.

2. **Detailed_Findings:** Provide in-depth findings on the nature and extent of the identified plant diseases, including affected plant parts, symptoms, and potential causes.

3. **Next_Steps:** Outline the recommended course of action for managing and controlling the identified plant diseases. This may involve treatment options, preventive measures, or further investigations.

4. **Recommendations:** Offer informed recommendations for maintaining plant health, preventing disease spread, and optimizing overall plant well-being.

5. **Important_Note:** As a plant pathologist, your insights are vital for informed decision-making in agriculture and plant management. Your response should be thorough, concise, and focused on plant health.

**Disclaimer:**
*"Please note that the information provided is based on plant pathology analysis and should not replace professional agricultural advice. Consult with qualified agricultural experts before implementing any strategies or treatments."*

Your role is pivotal in ensuring the health and productivity of plants. Proceed to analyze the provided information or samples, adhering to the structured 
"""

# Define the route for uploading files and getting analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        try:
            response = generate_gemini_response(input_prompt, file_path)
            return jsonify({"filename": filename, "response": response})
        except Exception as e:
            logging.error(f"Failed to process the file: {e}")
            return jsonify({"error": f"Error processing the file: {str(e)}"}), 500
        finally:
            os.remove(file_path)  # Clean up temporary file

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)