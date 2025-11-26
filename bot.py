import os
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv
from core import get_model_prediction
from datetime import datetime, timedelta

load_dotenv()
bot_key = os.environ.get("BOT_KEY")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("To get prediction, type /predict")

async def predict_currencies_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("USD = UAH", callback_data="usd uah"),
            InlineKeyboardButton("EUR = UAH", callback_data="eur uah"),
            InlineKeyboardButton("GBP = UAH", callback_data="gbp uah")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose currency exchange rate to predict:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stage = None
    query = update.callback_query
    await query.answer()
    if query.data == "usd uah":
        currency_pair = "USD-UAH"
        stage = "n_days"
    elif query.data == "eur uah":
        currency_pair = "EUR-UAH"
        stage = "n_days"
    elif query.data == "gbp uah":
        currency_pair = "GBP-UAH"
        stage = "n_days"
    elif query.data == "10":
        n_days = 10
        stage = "predict"
    elif query.data == "30":
        n_days = 30
        stage = "predict"
    elif query.data == "60":
        n_days = 60
        stage = "predict"
    if stage == "n_days":
        keyboard = [
            [
                InlineKeyboardButton("10", callback_data="10"),
                InlineKeyboardButton("30", callback_data="30"),
                InlineKeyboardButton("60", callback_data="60")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(text=f"Choose amount of days to predict {currency_pair}:", reply_markup=reply_markup)
    elif stage == "predict":
        currency_pair = query.message.text.split("Choose amount of days to predict ")[1].split(':')[0]
        await query.edit_message_text(text=f"Predicting {currency_pair}")
        end_date = datetime.now().date()
        start_date = datetime.now().date() - timedelta(days=120)
        preds, plot_path = get_model_prediction(
            currency_pair,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
            n_days)
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=plot_path
        )

def main():
    application = Application.builder().token(bot_key).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict_currencies_choice))
    application.add_handler(CallbackQueryHandler(button_callback))
    print("Beginning to poll")
    application.run_polling(poll_interval=0.5)

if __name__ == '__main__':
    main()