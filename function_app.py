import logging
import tensorflow as tf
import numpy as np
import os
import tempfile
from datetime import datetime, timedelta
from tensorflow.keras.preprocessing import image
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
import azure.functions as func
from twilio.rest import Client

# Connection string for Azure Blob Storage (store securely in Application Settings)
BLOB_CONNECTION_STRING = os.getenv("iotlabintrusion_STORAGE")
MODEL_CONTAINER_NAME = "models"  # Your blob container name
MODEL_BLOB_NAME = "irmodel.keras"  # The name of the blob containing the model

# Twilio credentials (store securely in Application Settings)
TWILIO_ACCOUNT_SID = 'AC60477f1e3fab56150ebfe3e4e4c2f108'
TWILIO_AUTH_TOKEN = 'a783b6150de99b8ba8dfac2f00935ba6'
TWILIO_WHATSAPP_FROM = "whatsapp:+14155238886"  # Twilio's WhatsApp sandbox number

# Function to send WhatsApp messages with optional media
def send_whatsapp_message(body, recipients, media_url=None):
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        for recipient in recipients:
            message = client.messages.create(
                from_=TWILIO_WHATSAPP_FROM,
                body=body,
                to=recipient,
                media_url=media_url  # Attach the media URL if provided
            )
            logging.info(f"WhatsApp message sent to {recipient}: SID {message.sid}")
    except Exception as e:
        logging.error(f"Failed to send WhatsApp message: {str(e)}")

# Load the model from blob storage dynamically
def download_model_from_blob():
    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        
        # Reference the container and blob
        container_client = blob_service_client.get_container_client(MODEL_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(MODEL_BLOB_NAME)
        
        # Download the blob content to a local file
        model_path = os.path.join(tempfile.gettempdir(), "irmodel.keras")
        with open(model_path, "wb") as model_file:
            model_file.write(blob_client.download_blob().readall())
        
        # Load the model into TensorFlow
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        logging.error(f"Error downloading model from Blob: {str(e)}")
        return None

# Generate a SAS URL for the blob using the connection string
def generate_blob_sas_url(container_name, blob_name):
    try:
        # Initialize the BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        
        # Generate the SAS token
        sas_token = generate_blob_sas(
            account_name=blob_service_client.account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)  # Valid for 1 hour
        )
        
        # Construct the full blob URL with the SAS token
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
        return blob_url
    except Exception as e:
        logging.error(f"Error generating SAS URL: {str(e)}")
        return None

# Initialize model (initial load when the function starts)
model = download_model_from_blob()

# Define the image preparation function
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale
    return img_array

# Blob Trigger Function
app = func.FunctionApp()

@app.blob_trigger(arg_name="myblob", path="image-upload/{name}", connection="iotlabintrusion_STORAGE")
def blob_trigger_iot(myblob: func.InputStream):
    try:
        # Log the blob name and size
        logging.info(f"Python blob trigger function processed blob"
                     f"Name: {myblob.name}, "
                     f"Blob Size: {myblob.length} bytes")
        
        # Extract the file name from the blob path
        file_name = myblob.name.split("/")[-1]
        
        # Sanitize the file name (if necessary)
        sanitized_file_name = file_name.replace("/", "_").replace(" ", "_")
        sanitized_file_name = sanitized_file_name[:255]  # Truncate if necessary
        
        # Create a dynamic path under the temp directory
        temp_path = os.path.join(tempfile.gettempdir(), sanitized_file_name)
        
        # Write the blob content to a temporary file
        with open(temp_path, "wb") as f:
            f.write(myblob.read())
        
        # Prepare the image for prediction
        prepared_image = prepare_image(temp_path)
        
        # Re-load the model if it wasn't loaded before or if the function is scaling
        global model
        if model is None:
            model = download_model_from_blob()
        
        # Make a prediction using the model
        prediction = model.predict(prepared_image)
        confidence = prediction[0][0]
        predicted_class = 'Class 1 (Positive)' if confidence > 0.5 else 'Class 0 (Negative)'

        # Log the result
        logging.info(f"Processed image: {myblob.name} - Prediction: {predicted_class} - Confidence: {confidence:.4f}")
        
        # Generate the SAS URL for the blob
        sas_url = generate_blob_sas_url("image-upload", file_name)
        
        if not sas_url:
            logging.error("Failed to generate SAS URL for the image blob")
            return

        # Log the generated SAS URL
        logging.info(f"Generated SAS URL: {sas_url}")

        # Define multiple recipient numbers
        recipients = [
            "whatsapp:+491621561715",  # Replace with the first recipient's WhatsApp number
            "whatsapp:+4915213037612",
            "whatsapp:+4915560768424",
            "whatsapp:+919904518345",  # Add other recipients here
        ]
        
        # Send a WhatsApp message with the prediction result and attach the image
        message_body = (f"Processed image: {myblob.name}\n"
                        f"Prediction: {predicted_class}\n"
                        f"Confidence: {confidence:.4f}")
        send_whatsapp_message(message_body, recipients, media_url=[sas_url])  # Pass the SAS URL as media

        # Optionally remove the temp file
        os.remove(temp_path)
    
    except Exception as e:
        logging.error(f"Error processing blob: {myblob.name}, error: {str(e)}")
