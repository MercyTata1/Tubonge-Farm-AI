# 🌾 Tubonge Farm AI Assistant

Tubonge Farm AI is a web-based assistant designed to empower smallholder farmers across Kenya with timely, localized agricultural advice powered by artificial intelligence.

## 🚀 Features

- 📸 Diagnose crop diseases from plant images (3 leaves + 1 whole plant)
- 💬 Ask farming questions and receive AI-generated advice from Gemini
- 🌦 Get localized weather updates with fallback estimates from AI
- 💰 Check market prices for maize, tomatoes, beans (with AI fallback)
- 🌐 Accessible via web on mobile or desktop, 24/7

## 🤖 Powered By

- [Streamlit](https://streamlit.io/) for interactive web UI
- [TensorFlow Lite](https://www.tensorflow.org/lite) for plant disease detection
- [Google Gemini Pro](https://makersuite.google.com/) for AI Q&A and knowledge fallback
- [Open-Meteo API](https://open-meteo.com/) for real-time weather data

## 📁 Folder Structure

```
tubonge-farm-ai/
├── app.py                  # Main Streamlit application
├── requirements.txt        # List of dependencies
├── models/
│   ├── crop_disease_model.tflite
│   └── class_mapping.json
```

## 🧠 Using the Gemini AI Key

To run this app, you must provide your Google AI Studio key.

1. Get your free key from: https://makersuite.google.com/app/apikey
2. Add it to your environment:
   - On Render/Streamlit: Set `GEMINI_API_KEY` as an environment variable
   - Or in Colab: Use `os.environ["GEMINI_API_KEY"] = "your-key"`

## 💻 How to Run

### Locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

### In Colab:
1. Upload all files
2. Use localtunnel:
```bash
!streamlit run app.py & npx localtunnel --port 8501
```
View here: https://tubonge-farm-ai-pzwg7j3b3szuuwlggnfnwa.streamlit.app/


## 🌍 Impact

- Supports Kenyan smallholders with AI-powered, accessible farming tools
- Promotes climate-smart agriculture and real-time problem-solving
- Designed to scale across Africa, farmer-first
