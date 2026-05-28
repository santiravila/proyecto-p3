
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# Función para el comando /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Soy proga_3_bot y ya estoy vivo.")

# Función para repetir lo que escribas
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(f"Recibí esto: {update.message.text}") #el bot ya esta, pero cuando terminemos la otra parte toca meter un RAG para que reconozca palabras especificas y pueda ejecutar comandos   
    
if __name__ == '__main__':
    
    app = ApplicationBuilder().token("8492808532:AAEL4mGGB_M6Of2qx6NCx5DfisuSbm6hnfw").build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), echo))

    print("--- El bot está encendido y esperando mensajes ---")
    app.run_polling()
 
''''
    python3 -m venv mi_bot_env
    source mi_bot_env/bin/activate
    pip install python-telegram-bot
    
    Si no te deja usar el fichero de telegram pon esto en el bash
    
    token del bot: 8492808532:AAEL4mGGB_M6Of2qx6NCx5DfisuSbm6hnfw
      
'''