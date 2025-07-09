from .. import loader, utils
import logging
import aiohttp

logger = logging.getLogger(__name__)

@loader.tds
class MistralAuto(loader.Module):
    """ Автоответчик в ЛС через Mistral AI с поддержкой памяти"""
    strings = {"name": "MistralAuto"}

    def __init__(self):
        self.dialogues = {}  # user_id: [messages]
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "mistral_key", "",
                lambda: "🔑 API-ключ от Mistral (начинается с Caf...)"
            ),
            loader.ConfigValue(
                "enabled", True,
                lambda: "🟢 Включить автоответ в ЛС"
            ),
            loader.ConfigValue(
                "system_prompt", "Ты дружелюбный и умный помощник. Отвечай кратко и по делу.",
                lambda: "🧠 Системный промт (инструкция для ИИ)"
            ),
            loader.ConfigValue(
                "use_memory", True,
                lambda: "🧠 Использовать память (историю диалога)"
            ),
            loader.ConfigValue(
                "max_history", 10,
                lambda: "📦 Максимум сообщений в истории (для каждого юзера)"
            ),
        )

    async def watcher(self, message):
        if not self.config["enabled"]:
            return
        if not message.is_private or message.out or not message.text:
            return

        await self.handle_ai(message)

    async def handle_ai(self, message):
        api_key = self.config["mistral_key"]
        if not api_key:
            await message.reply("❌ Укажи API ключ через `.config MistralAuto`")
            return

        uid = str(message.sender_id)
        user_text = message.text

        # Формируем историю диалога
        messages = []
        if self.config["use_memory"]:
            if uid not in self.dialogues:
                self.dialogues[uid] = []
            self.dialogues[uid].append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": self.config["system_prompt"]}] + self.dialogues[uid]
        else:
            messages = [
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": user_text}
            ]

        # Обрезка истории
        max_len = self.config["max_history"]
        if self.config["use_memory"]:
            self.dialogues[uid] = self.dialogues[uid][-max_len * 2:]  # user/assistant чередуются

        # Запрос к Mistral API
        try:
            reply = await self.ask_mistral(messages, api_key)
            if self.config["use_memory"]:
                self.dialogues[uid].append({"role": "assistant", "content": reply})
            await message.reply(reply)
        except Exception as e:
            logger.error("Mistral API error: %s", e)
            await message.reply("⚠️ Ошибка при обращении к Mistral API.")

    async def ask_mistral(self, messages, api_key):
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-medium",
            "messages": messages,
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data, headers=headers) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()

    @loader.command()
    async def mistral(self, message):
        """<текст> — ручной запрос к Mistral"""
        text = utils.get_args_raw(message)
        if not text:
            await utils.answer(message, "📌 Использование: `.mistral твой вопрос`")
            return

        api_key = self.config["mistral_key"]
        if not api_key:
            await utils.answer(message, "❌ Укажи API ключ через `.config MistralAuto`")
            return

        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": text}
        ]

        try:
            reply = await self.ask_mistral(messages, api_key)
            await utils.answer(message, reply)
        except Exception as e:
            logger.error("Mistral command error: %s", e)
            await utils.answer(message, "⚠️ Ошибка при запросе к Mistral.")