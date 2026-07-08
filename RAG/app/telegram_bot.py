import logging
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
import requests

# Configuración de logs
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Estados para el flujo de reporte
ESPECIE, UBICACION, FOTO = range(3)

# URL de tu API de FastAPI
API_URL = "http://localhost:8000/ask"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐾 ¡Hola! Soy el asistente de Fauna Silvestre BQ.\n\n"
        "Puedo ayudarte con dos cosas:\n"
        "1. Hazme preguntas sobre animales de la ciudad.\n"
        "2. Usa /reportar para informar sobre un avistamiento o animal en peligro."
    )

# --- FLUJO DE PREGUNTAS (RAG) ---
async def handle_questions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pregunta = update.message.text
    await update.message.reply_chat_action("typing")
    
    # Llamada a tu API de FastAPI
    try:
        response = requests.post(API_URL, json={"question": pregunta})
        data = response.json()
        respuesta = data["answer"]
        await update.message.reply_text(respuesta)
    except Exception as e:
        await update.message.reply_text("Error al conectar con el servidor RAG.")

# --- FLUJO DE REPORTE (NUEVO) ---
async def iniciar_reporte(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("¿Qué animal estás reportando? (Ej: Una iguana, una titi cabeciblanco...)")
    return ESPECIE

async def capturar_especie(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['especie'] = update.message.text
    await update.message.reply_text("¿En qué barrio o sector de Barranquilla te encuentras?")
    return UBICACION

async def capturar_ubicacion(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data['ubicacion'] = update.message.text
    await update.message.reply_text("Por favor, envía una foto del animal (o envía /saltar)")
    return FOTO

async def finalizar_reporte(update: Update, context: ContextTypes.DEFAULT_TYPE):
    especie = context.user_data['especie']
    ubicacion = context.user_data['ubicacion']
    
    # Aquí podrías enviar estos datos a una base de datos real a través de FastAPI
    resumen = f"✅ Reporte Recibido:\nAnimal: {especie}\nUbicación: {ubicacion}\n\nGracias por ayudar a proteger nuestra fauna."
    await update.message.reply_text(resumen)
    return ConversationHandler.END

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Reporte cancelado.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

if __name__ == '__main__':
    from dotenv import load_dotenv
    import os
    load_dotenv()
    
    app = Application.builder().token(os.getenv("TELEGRAM_TOKEN")).build()

    # Manejador de la conversación de reporte
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('reportar', iniciar_reporte)],
        states={
            ESPECIE: [MessageHandler(filters.TEXT & ~filters.COMMAND, capturar_especie)],
            UBICACION: [MessageHandler(filters.TEXT & ~filters.COMMAND, capturar_ubicacion)],
            FOTO: [MessageHandler(filters.PHOTO | filters.COMMAND, finalizar_reporte)],
        },
        fallbacks=[CommandHandler('cancelar', cancelar)],
    )

    app.add_handler(conv_handler)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_questions))
    app.add_handler(CommandHandler("start", start))

    print("🤖 Bot de Telegram en marcha...")
    app.run_polling()