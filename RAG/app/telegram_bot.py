import logging
import os
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardRemove
from telegram.ext import Application, MessageHandler, filters, ContextTypes, ConversationHandler
import requests

load_dotenv()

# Configuración de logs
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Estados para el flujo de reporte
ESPECIE, UBICACION, FOTO = range(3)

# URL de tu API de FastAPI
API_URL = "http://localhost:8000/ask"

# --- FUNCION DE INICIO (Ahora responde a "Hola", "Inicio", etc.) ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐾 ¡Hola! Soy el asistente de Fauna Silvestre BQ.\n\n"
        "Puedo ayudarte con dos cosas:\n"
        "1. Hazme preguntas sobre animales de la ciudad.\n"
        "2. Escribe 'reportar' para informar sobre un avistamiento o animal en peligro."
    )

# --- FLUJO DE REPORTE ---
async def iniciar_reporte(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("¿Qué animal estás reportando? (Ej: Una iguana, una titi cabeciblanco...)")
    return ESPECIE

async def capturar_especie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['especie'] = update.message.text
    await update.message.reply_text("¿En qué barrio o sector de Barranquilla te encuentras?")
    return UBICACION

async def capturar_ubicacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['ubicacion'] = update.message.text
    await update.message.reply_text("Por favor, envía una foto del animal (o escribe 'saltar')")
    return FOTO

async def finalizar_reporte(update: Update, context: ContextTypes.DEFAULT_TYPE):
    especie = context.user_data.get('especie', 'Desconocido')
    ubicacion = context.user_data.get('ubicacion', 'Desconocida')
    
    resumen = f" Reporte Recibido:\nAnimal: {especie}\nUbicación: {ubicacion}\n\nGracias por ayudar a proteger nuestra fauna."
    await update.message.reply_text(resumen)
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Reporte cancelado.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# --- FLUJO DE PREGUNTAS (RAG) ---
async def handle_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pregunta = update.message.text
    
    # Si el usuario escribe algo muy genérico como "hola", saludamos en lugar de ir al RAG
    if pregunta.lower() in ["hola", "buenas", "inicio", "start"]:
        await start(update, context)
        return

    await update.message.reply_chat_action("typing")
    
    try:
        response = requests.post(API_URL, json={"question": pregunta})
        data = response.json()
        respuesta = data["answer"]
        await update.message.reply_text(respuesta)
    except Exception as e:
        await update.message.reply_text("Error al conectar con el servidor RAG.")

if __name__ == '__main__':
    app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

    # --- CONFIGURACIÓN DEL MANEJADOR DE CONVERSACIÓN ---
    conv_handler = ConversationHandler(
        # Ahora entra con la palabra "reportar" o "reporte" (ignorando mayúsculas)
        entry_points=[MessageHandler(filters.Regex(r'(?i)^(reportar|reporte)$'), iniciar_reporte)],
        states={
            ESPECIE: [MessageHandler(filters.TEXT & ~filters.COMMAND, capturar_especie)],
            UBICACION: [MessageHandler(filters.TEXT & ~filters.COMMAND, capturar_ubicacion)],
            # Acepta foto o la palabra "saltar"
            FOTO: [
                MessageHandler(filters.PHOTO, finalizar_reporte),
                MessageHandler(filters.Regex(r'(?i)^saltar$'), finalizar_reporte)
            ],
        },
        # Ahora sale con la palabra "cancelar"
        fallbacks=[MessageHandler(filters.Regex(r'(?i)^cancelar$'), cancelar)],
    )


    # 1. El conversador tiene prioridad (para detectar "reportar")
    app.add_handler(conv_handler)
    
    # 2. El manejador de preguntas RAG captura todo lo demás
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_questions))

    print(" Bot de Telegram en marcha (sin comandos /)...")
    app.run_polling()
