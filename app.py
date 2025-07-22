import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from bs4 import BeautifulSoup
import json
from geopy.geocoders import Nominatim
import os
import google.generativeai as genai

# === Set your Gemini API key ===
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="Tubonge Farm AI", layout="wide", page_icon="🌾")

class TubongeAI:
    def __init__(self):
        self.model = tf.lite.Interpreter('crop_disease_model.tflite')
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()

        self.geolocator = Nominatim(user_agent="tubonge_app")
        self.gemini_model = genai.GenerativeModel('gemini-pro')

        if 'history' not in st.session_state:
            st.session_state.history = []
            st.session_state.location = None

    def get_weather(self, location):
        try:
            geo = self.geolocator.geocode(location)
            lat, lon = geo.latitude, geo.longitude
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,precipitation"
            res = requests.get(url, timeout=10).json()
            temp = res["current"]["temperature_2m"]
            rain = res["current"]["precipitation"]
            return {"temperature": temp, "rain": rain, "location": location}
        except:
            try:
                prompt = f"What is the typical weather in {location}, Kenya during this time of year?"
                response = self.gemini_model.generate_content(prompt)
                return {"error": "Live data unavailable. Here's an estimate:", "ai_estimate": response.text.strip()}
            except:
                return {"error": "Sorry, I couldn’t retrieve weather data. Try again later."}

    def diagnose_image(self, images):
        try:
            if len(images) < 4:
                return {'status': 'incomplete', 'message': 'Please upload 4 images (3 leaves + 1 whole plant)'}
            predictions = []
            for img_file in images[:4]:
                img = Image.open(img_file).resize((224, 224))
                img_array = np.expand_dims(np.array(img) / 255.0, 0).astype(np.float32)
                self.model.set_tensor(self.input_details[0]['index'], img_array)
                self.model.invoke()
                pred = self.model.get_tensor(self.output_details[0]['index'])[0]
                predictions.append(pred)
            avg_pred = np.mean(predictions, axis=0)
            class_index = np.argmax(avg_pred)
            confidence = np.max(avg_pred)
            with open("class_mapping.json") as f:
                labels = json.load(f)
            pest = labels[str(class_index)]
            return {
                'status': 'success',
                'pest': pest,
                'confidence': f"{confidence:.0%}",
                'solution': self._get_knowledge(pest),
                'prevention': self._get_prevention(pest)
            }
        except Exception as e:
            return {'status': 'error', 'message': f"Analysis failed: {str(e)}"}

    def _get_knowledge(self, topic):
        try:
            prompt = (
                f"Provide a 3-sentence agricultural solution for {topic} in Kenya. "
                "Use localized advice and cite a trusted source if possible."
            )
            response = self.gemini_model.generate_content(prompt)
            return response.text.strip()
        except:
            try:
                url = f"https://www.infonet-biovision.org/Search?search={topic}"
                res = requests.get(url, timeout=5)
                soup = BeautifulSoup(res.text, 'html.parser')
                result = soup.select_one(".search-results li")
                if result:
                    return f"{result.text[:300]}... (Source: Infonet)"
            except:
                return f"Sorry, I couldn’t find reliable info on '{topic}'. Please try rephrasing your question."

    def get_market_prices(self, crop, location="Nairobi"):
        price_table = {
            "maize": {"Nairobi": 4500, "Kisumu": 4200},
            "tomatoes": {"Nairobi": 80, "Kisumu": 70},
            "beans": {"Nairobi": 150, "Kisumu": 130}
        }
        price = price_table.get(crop, {}).get(location)
        if price:
            return price
        else:
            try:
                prompt = f"What is the estimated market price for {crop} in {location}, Kenya this month?"
                response = self.gemini_model.generate_content(prompt)
                return f"AI Estimate: {response.text.strip()}"
            except:
                return "No data or estimate available."

    def _get_prevention(self, pest):
        prevention_db = {
            "fall_armyworm": "Rotate crops with legumes, use pheromone traps",
            "aphids": "Introduce ladybugs, spray neem oil weekly",
            "leaf_blight": "Use certified seeds, avoid overcrowding"
        }
        return prevention_db.get(pest.lower(), "Practice crop rotation and good field hygiene.")

    def ask_gemini(self, question):
        try:
            response = self.gemini_model.generate_content(
                f"You are a local agricultural advisor in Kenya. Answer this farmer's question simply and clearly: {question}. "
                "Mention local context or practical actions where possible."
            )
            return response.text.strip()
        except Exception as e:
            return f"Sorry, AI could not answer your question: {str(e)}"

def main():
    ai = TubongeAI()
    st.title("🌾 Tubonge Farm AI Assistant")
    st.markdown("Upload plant images, ask farming questions, and get local weather/market info.")

    location = st.text_input("📍 Enter your location (e.g. Busia, Nairobi):")
    if location:
        st.session_state.location = location

    tab1, tab2, tab3 = st.tabs(["🧪 Diagnose", "💬 Advice", "🌤 Market & Weather"])

    with tab1:
        files = st.file_uploader("Upload 4 images (3 leaves + 1 plant):", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        if files:
            result = ai.diagnose_image(files)
            if result['status'] == 'success':
                st.success(f"✅ Detected: {result['pest']} ({result['confidence']} confidence)")
                st.write(f"**Solution:** {result['solution']}")
                st.write(f"**Prevention:** {result['prevention']}")
                cols = st.columns(4)
                for i, img in enumerate(files[:4]):
                    cols[i].image(Image.open(img), caption=f"Image {i+1}")
            else:
                st.error(result['message'])

    with tab2:
        question = st.text_input("Ask a question about farming (e.g. how to grow tomatoes):")
        if question:
            answer = ai.ask_gemini(question)
            st.info(answer)

    with tab3:
        if st.session_state.location:
            weather = ai.get_weather(st.session_state.location)
            if "temperature" in weather:
                st.metric("🌡 Temperature", f"{weather['temperature']}°C")
                st.metric("🌧 Precipitation", f"{weather['rain']} mm")
            elif "ai_estimate" in weather:
                st.warning(weather["error"])
                st.info(weather["ai_estimate"])
            else:
                st.error(weather["error"])

            crop = st.selectbox("Select crop", ["maize", "tomatoes", "beans"])
            price = ai.get_market_prices(crop, st.session_state.location)
            if isinstance(price, str) and price.startswith("AI Estimate"):
                st.warning("No official data. Here's an AI-generated estimate:")
                st.info(price)
            elif isinstance(price, str):
                st.error(price)
            else:
                st.metric(f"{crop.title()} Price", f"KSh {price}")

if __name__ == "__main__":
    main()
