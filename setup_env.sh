#!/bin/bash

# Nombre del entorno virtual
VENV_NAME="venv"

echo "Iniciando configuración del entorno para el proyecto de Computer Vision..."

# 1. Crear el entorno virtual
echo "Creando entorno virtual en './$VENV_NAME'..."
python -m venv $VENV_NAME

if [ $? -eq 0 ]; then
    echo "✅ Entorno virtual creado con éxito."
    
    # 2. Determinar el script de activación (Windows vs Linux/macOS)
    if [ -d "$VENV_NAME/Scripts" ]; then
        ACTIVATE_PATH="$VENV_NAME/Scripts/activate"
    else
        ACTIVATE_PATH="$VENV_NAME/bin/activate"
    fi

    # 3. Activar el entorno e instalar dependencias
    echo "Activando el entorno..."
    source "$ACTIVATE_PATH"

    echo "Actualizando pip e instalando dependencias desde requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt

    if [ $? -eq 0 ]; then
        echo "✅ Instalación completada correctamente."
        echo "Para usar el entorno, ejecuta: source $ACTIVATE_PATH"
    else
        echo "❌ Error al instalar las dependencias."
        exit 1
    fi
else
    echo "❌ Error al crear el entorno virtual. Asegúrate de tener Python instalado y accesible."
    exit 1
fi
