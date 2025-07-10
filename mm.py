from .. import loader, utils
import logging
import aiohttp

logger = logging.getLogger(__name__)

@loader.tds
class MistralAuto(loader.Module):
    """Автоответчик в ЛС через Mistral AI с памятью, блокировкой и ручными запросами"""
    strings = {"name": "MistralAuto"}

    def __init__(self):
        self.dialogues = {}
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "mistral_key", "",
                lambda: "🔑 API-ключ от Mistral (начинается с Caf...)"
            ),
            loader.ConfigValue(
                "enabled", True,
                lambda: "🟢 Включить автоответ глобально"
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
                lambda: "📦 Максимум сообщений в истории"
            ),
            loader.ConfigValue(
                "blocked_users", [],
                lambda: "🚫 Список отключённых пользователей"
            ),
        )

    async def watcher(self, message):
        if not self.config["enabled"]:
            return
        if not message.is_private or message.out or not message.text:
            return
        if message.sender_id in self.config["blocked_users"]:
            return

        await self.handle_ai(message)

    async def handle_ai(self, message):
        api_key = self.config["mistral_key"]
        if not api_key:
            await message.reply("❌ Укажи API ключ через `.config MistralAuto`")
            return

        uid = str(message.sender_id)
        user_text = message.text
        messages = []

        if self.config["use_memory"]:
            if uid not in self.dialogues:
                self.dialogues[uid] = []
            self.dialogues[uid].append({"role": "user", "content": user_text})
            messages = [{"role": "system", "content": self.config["system_prompt"]}] + self.dialogues[uid]
            self.dialogues[uid] = self.dialogues[uid][-self.config["max_history"] * 2:]
        else:
            messages = [
                {"role": "system", "content": self.config["system_prompt"]},
                {"role": "user", "content": user_text}
            ]

        try:
            reply = await self.ask_mistral(messages, api_key)
            if self.config["use_memory"]:
                self.dialogues[uid].append({"role": "assistant", "content": reply})
            await message.reply(reply)
        except Exception as e:
            await message.reply(f"⚠️ Ошибка Mistral: {e}")

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
                if "choices" not in result:
                    raise Exception(result.get("error", "неизвестная ошибка"))
                return result["choices"][0]["message"]["content"].strip()

    @loader.command()
    async def mistral(self, message):
        """<вопрос> — ручной запрос к Mistral"""
        text = utils.get_args_raw(message)
        if not text:
            return await utils.answer(message, "📌 `.mistral твой вопрос`")

        key = self.config["mistral_key"]
        if not key:
            return await utils.answer(message, "❌ Укажи API ключ через `.config MistralAuto`")

        messages = [
            {"role": "system", "content": self.config["system_prompt"]},
            {"role": "user", "content": text}
        ]

        try:
            reply = await self.ask_mistral(messages, key)
            await utils.answer(message, reply)
        except Exception as e:
            await utils.answer(message, f"⚠️ Ошибка Mistral: {e}")

    @loader.command()
    async def mistraltoggle(self, message):
        """Включить/отключить автоответ глобально"""
        current = self.config["enabled"]
        self.config["enabled"] = not current
        await utils.answer(message, f"✅ Автоответ: {'включён' if not current else 'отключён'}")

    @loader.command()
    async def mistralblock(self, message):
        """<@ или id> — отключить автоответ для пользователя"""
        user = await self._get_user_id(message)
        if user is None:
            return await utils.answer(message, "❌ Не удалось определить пользователя.")
        if user in self.config["blocked_users"]:
            return await utils.answer(message, "⚠️ Уже в списке.")
        self.config["blocked_users"].append(user)
        await utils.answer(message, f"🚫 Пользователь `{user}` заблокирован для автоответов.")

    @loader.command()
    async def mistralunblock(self, message):
        """<@ или id> — включить автоответ для пользователя"""
        user = await self._get_user_id(message)
        if user is None:
            return await utils.answer(message, "❌ Не удалось определить пользователя.")
        if user not in self.config["blocked_users"]:
            return await utils.answer(message, "⚠️ Пользователь не в списке.")
        self.config["blocked_users"].remove(user)
        await utils.answer(message, f"✅ Пользователь `{user}` разблокирован.")

    async def _get_user_id(self, message):
        args = utils.get_args_raw(message)
        if not args and message.reply_to:
            reply = await message.get_reply_message()
            return reply.sender_id
        if args.isdigit():
            return int(args)
        try:
            entity = await message.client.get_entity(args)
            return entity.id
        except Exception:
            return None