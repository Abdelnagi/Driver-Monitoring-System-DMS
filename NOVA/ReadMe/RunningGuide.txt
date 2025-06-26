

# 🚘 Driver Assistant Module – Setup & Run Guide

This is a real-time web-based driver assistant system that runs in two modes:  
- **Monitoring Mode:** Monitors driver's state and behavior using sensor data and triggers alerts when fatigue, distractions, or dangerous activity is detected.  
- **Assistant Mode:** Voice-activated assistant using LLaMA 3.2 (via Ollama) for general help, navigation, and emergency response.

---

## 📦 Step-by-Step Installation Guide

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/farahahmed09/DMS_EECE25_CU.git
cd DMS_EECE25_CU
```

---

### ✅ 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:**  
> If you encounter errors for specific libraries (e.g., `sounddevice`, `eel`, or `edge-tts`), install them manually:
```bash
pip install eel sounddevice scipy pydub requests geopy groq edge-tts
```

---

### ✅ 3. Install Ollama (for LLaMA Assistant)
1. Visit: https://ollama.com/download/windows  
2. Download and install the Ollama Windows app.
3. Open a terminal and run:
```bash
ollama run llama3
```

> **⚠️ Memory Warning**  
> If Ollama shows an "insufficient RAM" error:
> - Close all browser tabs and apps  
> - Restart your laptop  
> - Then try again:  
> ```bash
> ollama run llama3
> ```

---

### ✅ 4. Configure `config.py`
Open the `config.py` file and fill in the following:

> Don't leave any of these fields blank — they're critical for emergency notification and transcription.

---

### ✅ 5. Run the App
```bash
python main.py
```

This will:
- Launch the assistant web dashboard in your browser
- Start monitoring driver state via the live JSON file
- Respond to wake word or mic press



