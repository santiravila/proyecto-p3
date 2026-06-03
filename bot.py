from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from rag_test import ask

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.effective_message
    if not message:
        return

    await message.reply_text(
        "Soy tu Asistente Documental RAG local. \n\n"
        "Preguntame sobre los documentos que he memorizado."
    )

async def responder_rag(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.effective_message
    if not message:
        return

    pregunta_usuario = message.text
    if pregunta_usuario is None:
        return

    # Avisar que se esta procesando
    mensaje_espera = await message.reply_text("Buscando en la base vectorial y generando respuesta... ⏳ (Esto puede tomar hasta un minuto)")
    
    try:
        # Llamar RAG
        respuesta_ia = ask(pregunta_usuario)
        
        # eliminar/editar mensaje de espera
        await message.reply_text(respuesta_ia)
    
        chat_id = getattr(message.chat, 'id', None) or getattr(update.effective_chat, 'id', None)
        if mensaje_espera and chat_id:
            await context.bot.delete_message(chat_id=chat_id, message_id=mensaje_espera.message_id)
        
    except Exception as e:
        await message.reply_text("Hubo un error al procesar tu pregunta. Revisa la consola del servidor.")
        print(f"Error en RAG: {e}")

if __name__ == '__main__':
    # TOKEN DE TELEGRAM
    TOKEN = "8997695689:AAFZAHbLa6fMIRvgFQAHc1e31yU5El4ui8E"
    
    # Dar 120 segundos para pensar
    app = ApplicationBuilder().token(TOKEN).read_timeout(120).write_timeout(120).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), responder_rag))

    print("--- 🤖 El bot RAG está ENCENDIDO y escuchando en Telegram ---")
    app.run_polling()