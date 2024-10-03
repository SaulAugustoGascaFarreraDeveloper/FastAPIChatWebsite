FROM python:3.9

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos necesarios al contenedor
COPY . .

# Instala las dependencias de tu aplicación
RUN pip install -r requirements.txt

# Expone el puerto en el que tu aplicación escucha
EXPOSE 8000

# Comando para ejecutar tu aplicación
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]