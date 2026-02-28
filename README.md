⚽ Predictor LaLiga

Predicción de partidos de LaLiga usando datos históricos y, opcionalmente, cuotas de apuestas para mejorar la precisión.

📋 Requisitos

Python 3.8 o superior

Comprueba tu versión de Python:

python --version

o, si usas Linux/macOS:

python3 --version
🐍 Instalación y configuración del entorno virtual

Usar un entorno virtual (venv) evita conflictos con otras librerías de Python.

1️⃣ Crear el entorno virtual

Desde la carpeta del proyecto:

python -m venv venv
2️⃣ Activar el entorno virtual
Linux / macOS
source venv/bin/activate
Windows (PowerShell)
venv\Scripts\Activate.ps1
Windows (CMD)
venv\Scripts\activate

✅ Cuando el entorno esté activo, verás algo así en la terminal:

(venv)
3️⃣ Instalar dependencias

Si el proyecto incluye un archivo requirements.txt:

pip install -r requirements.txt
🚀 Uso del predictor

⚠️ Asegúrate de que el entorno venv esté activado antes de ejecutar los siguientes comandos.

# 1. Entrenar el modelo
python predictor_laliga.py --train

# 2. Predecir con cuotas (más preciso)
python predictor_laliga.py --predict "Real Madrid" "FC Barcelona" --odds 1.80 3.50 4.20

# 3. Predecir sin cuotas (usa solo la forma)
python predictor_laliga.py --predict "Real Madrid" "Atletico de Madrid"

# 4. Modo interactivo
python predictor_laliga.py
🛑 Salir del entorno virtual

Cuando termines de usar el proyecto:

deactivate
🆘 Problemas comunes

python no se reconoce como comando
👉 Asegúrate de que Python esté correctamente instalado y añadido al PATH.

Error al activar el entorno en PowerShell
👉 Ejecuta PowerShell como administrador y luego:

Set-ExecutionPolicy RemoteSigned
📌 Notas

Siempre ejecuta el proyecto con el entorno virtual activado.

Si instalas nuevas librerías, recuerda actualizar requirements.txt.