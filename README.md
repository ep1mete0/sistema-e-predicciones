# ⚽ Predictor LaLiga

Predicción de partidos de LaLiga usando datos históricos y, opcionalmente, cuotas de apuestas para mejorar la precisión. El modelo aprende con el tiempo: puedes registrar resultados reales y reentrenar para mejorar la precisión progresivamente.

---

## 📋 Requisitos

Python 3.8 o superior

Comprueba tu versión de Python:

```bash
python --version
```

o, si usas Linux/macOS:

```bash
python3 --version
```

---

## 🐍 Instalación y configuración del entorno virtual

Usar un entorno virtual (`venv`) evita conflictos con otras librerías de Python.

### 1️⃣ Crear el entorno virtual

Desde la carpeta del proyecto:

```bash
python -m venv venv
```

### 2️⃣ Activar el entorno virtual

**Linux / macOS**
```bash
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```cmd
venv\Scripts\activate
```

✅ Cuando el entorno esté activo, verás algo así en la terminal:

```
(venv)
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## 🚀 Uso del predictor

> ⚠️ Asegúrate de que el entorno `venv` esté activado antes de ejecutar los siguientes comandos.

### Entrenar el modelo
```bash
python main.py --train
```

### Predecir un partido (con cuotas — más preciso)
```bash
python main.py --predict "Real Madrid" "FC Barcelona" --odds 1.80 3.50 4.20
```

### Predecir un partido (sin cuotas — usa solo la forma reciente)
```bash
python main.py --predict "Real Madrid" "Atletico de Madrid"
```

### Registrar el resultado real después de un partido
```bash
python main.py --resultado "Real Madrid" "FC Barcelona" H
```
Valores posibles: `H` (gana local) · `D` (empate) · `A` (gana visitante)

### Registrar resultado incluyendo los goles
```bash
python main.py --resultado "Real Madrid" "FC Barcelona" H --goles 2 1
```

### Ver el historial de partidos registrados
```bash
python main.py --historial
```

### Ver estadísticas del historial
```bash
python main.py --stats
```

### Modo interactivo (menú completo)
```bash
python main.py
```

---

## 🔄 Flujo de aprendizaje continuo

El modelo mejora con el tiempo siguiendo este ciclo:

```
Predecir partido → Esperar resultado → Registrar resultado → Reentrenar
```

Cada resultado registrado se guarda en `historial_partidos.json` y se incorpora automáticamente al siguiente entrenamiento. Cuantos más partidos registres, más preciso se vuelve el modelo.

Para añadir datos de una nueva temporada, agrega el archivo CSV a la lista `CSV_FILES` dentro del script y vuelve a entrenar:

```python
CSV_FILES = [
    'SP1_2021.csv',
    ...
    'SP1_2026.csv',  # ← nuevo archivo
]
```

```bash
python main.py --train
```

---

## 🛑 Salir del entorno virtual

Cuando termines de usar el proyecto:

```bash
deactivate
```

---

## 🆘 Problemas comunes

**`python` no se reconoce como comando**
👉 Asegúrate de que Python esté correctamente instalado y añadido al PATH.

**Error al activar el entorno en PowerShell**
👉 Ejecuta PowerShell como administrador y luego:

```powershell
Set-ExecutionPolicy RemoteSigned
```

---

## 📌 Notas

- Siempre ejecuta el proyecto con el entorno virtual activado.
- Si instalas nuevas librerías, recuerda actualizar `requirements.txt`.
- Los datos históricos de partidos se descargan desde [football-data.co.uk](https://www.football-data.co.uk/spainm.php).