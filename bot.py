import os
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from dotenv import load_dotenv
from core import get_model_prediction, get_history_rate, get_yf_data
from datetime import datetime, timedelta

load_dotenv()
bot_key = os.environ.get("BOT_KEY")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("To get prediction, type /predict. To get exchange rates for last 120 days, type /history_rate.")

async def predict_currencies_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("USD = UAH", callback_data="usd uah predict"),
            InlineKeyboardButton("EUR = UAH", callback_data="eur uah predict"),
            InlineKeyboardButton("GBP = UAH", callback_data="gbp uah predict")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose currency exchange rate to predict:", reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stage = None
    query = update.callback_query
    await query.answer()
    status = query.data.split(' ')
    if len(status) == 3:
        if status[0] == "usd":
            currency_pair = "USD-UAH"
        elif status[0] == "eur":
            currency_pair = "EUR-UAH"
        elif status[0] == "gbp":
            currency_pair = "GBP-UAH"
        if status[2] == "predict":
            stage = "n_days"
        else:
            stage = "history"
    else:
        n_days = int(status[0])
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
    elif stage == "history":
        end_date = datetime.now().date()
        start_date = datetime.now().date() - timedelta(days=120)
        plot_path = get_history_rate(
            currency_pair,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=plot_path
        )

async def history_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [
            InlineKeyboardButton("USD = UAH", callback_data="usd uah history"),
            InlineKeyboardButton("EUR = UAH", callback_data="eur uah history"),
            InlineKeyboardButton("GBP = UAH", callback_data="gbp uah history")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose currency exchange rate:", reply_markup=reply_markup)

async def health(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("The bot is working.")

async def current_rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    end_date = datetime.now().date()
    start_date = datetime.now().date() - timedelta(days=120)
    usd_rates = get_yf_data(start_date, end_date, "USDUAH=X")
    eur_rates = get_yf_data(start_date, end_date, "EURUAH=X")
    gbp_rates = get_yf_data(start_date, end_date, "GBPUAH=X")
    await update.message.reply_text(f"""Current exchange rates:
USD-UAH: {float(usd_rates.iloc[-1]['Close']):.2f}₴
EUR-UAH: {float(eur_rates.iloc[-1]['Close']):.2f}₴
GBP-UAH: {float(gbp_rates.iloc[-1]['Close']):.2f}₴""")

def main():
    application = Application.builder().token(bot_key).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict_currencies_choice))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(CommandHandler("history_rate", history_rate))
    application.add_handler(CommandHandler("health", health))
    application.add_handler(CommandHandler("current_rate", current_rate))
    print("Beginning to poll")
    application.run_polling(poll_interval=0.5)

if __name__ == '__main__':
    main()