import streamlit as st
import google.generativeai as genai
import tempfile
import os
from PIL import Image as PILImage
import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
import os
# --- SETUP ---


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

eczema_model = tf.keras.models.load_model("eczema_model_transfer_learning.h5")

# --- Streamlit UI ---

st.set_page_config(page_title="Eczema Classifier", layout="centered")
st.title("🧑‍⚕️ Eczema Classifier")

st.markdown("Upload a skin image, and our intelligent assistant will detect signs of **eczema**.")

uploaded_file = st.file_uploader("Upload a skin image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing image..."):

            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # Step 1: Validate it's a skin image using Gemini
                with open(tmp_path, "rb") as img_file:
                    image_data = img_file.read()

                validation_response = gemini_model.generate_content(
                    [
                        "Is this a photo of a human or a human skin image like face,hands, neck, legs,nails etc? Just answer Yes or No.",
                        {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    ]
                )

                is_skin = "yes" in validation_response.text.lower()

                if not is_skin:
                    st.warning("⚠️ This doesn't appear to be a human skin image. Please upload a valid skin image.")
                    os.remove(tmp_path)
                    st.stop()

                # Step 2: Run local CNN model
                img = PILImage.open(tmp_path).convert("RGB").resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                cnn_pred = eczema_model.predict(img_array)[0][0]
                cnn_result = "Eczema" if cnn_pred > 0.5 else "Not Eczema"

                # Step 3: Run Gemini model to cross-verify
                gemini_response = gemini_model.generate_content(
                    [
                        "Based on this image, classify the skin condition as:\n- Eczema\n- Not Eczema",
                        {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }
                    ]
                )

                gemini_text = gemini_response.text.lower()
                gemini_result = "Eczema" if "eczema" in gemini_text and "not eczema" not in gemini_text else "Not Eczema"

                # Step 4: Final decision (consensus or fallback)
                if cnn_result == gemini_result:
                    final_diagnosis = cnn_result
                else:
                    final_diagnosis = gemini_result  # fallback to Gemini

                # Step 5: Show final result to user
                st.success(f"✅ Diagnosis: **{final_diagnosis}**")

            except Exception as e:
                st.error("❌ Error processing image.")
                st.exception(e)
            finally:
                os.remove(tmp_path)
