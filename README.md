# 🏃 RBA — Running Biomechanics Analyzer

## Estructura de archivos
```
RunningAnalyzer/
├── app.py
├── requirements.txt
├── Procfile
├── .streamlit/config.toml
└── running_analyzer.py
```

## Correr localmente
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy en Railway (link permanente gratis)
1. Sube todos los archivos a GitHub (repositorio público)
2. Ve a https://railway.app → "Login with GitHub"
3. "New Project" → "Deploy from GitHub repo"
4. Selecciona tu repo → Railway detecta el Procfile solo
5. En 2-3 min tendrás tu link para compartir

## Ver en celular (WiFi local)
La terminal mostrará: `Network URL: http://192.168.x.x:8501`
Abre esa URL en el navegador de tu celular.
