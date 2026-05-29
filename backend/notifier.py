"""
WhatsApp fraud alert via Twilio Sandbox.

Reads credentials from env vars (dotenv-загружен в main.py):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_WHATSAPP_FROM     # e.g. "+14155238886" (Twilio Sandbox)
    GUARD_CALL_VICTIM_NAME   # имя того, кому звонят (для тела сообщения)
    GUARD_CALL_FAMILY_PHONE  # номер получателя, e.g. "+972537158722"

Если переменных нет — вызов превращается в no-op + лог, бэкенд не падает.
"""
import os
import asyncio


def _config():
    return {
        "sid":   os.getenv("TWILIO_ACCOUNT_SID", ""),
        "token": os.getenv("TWILIO_AUTH_TOKEN", ""),
        "from":  os.getenv("TWILIO_WHATSAPP_FROM", "+14155238886"),
        "name":  os.getenv("GUARD_CALL_VICTIM_NAME", "user"),
        "to":    os.getenv("GUARD_CALL_FAMILY_PHONE", ""),
    }


def _send_sync(victim_name: str, to_phone: str, matched_phrases: list) -> str:
    """Synchronous Twilio call — wrapped in executor by send_fraud_alert()."""
    from twilio.rest import Client

    cfg = _config()
    if not (cfg["sid"] and cfg["token"]):
        raise RuntimeError("Twilio credentials missing (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN)")

    phrases_text = ", ".join(matched_phrases) if matched_phrases else "—"
    body = (
        f"🚨 אזהרה! מתקבלת כעת שיחה חשודה אצל {victim_name}. "
        f"זוהו סימני הונאה: {phrases_text}. "
        f"התקשרו אליו/אליה מיד!"
    )

    client = Client(cfg["sid"], cfg["token"])
    msg = client.messages.create(
        body=body,
        from_=f"whatsapp:{cfg['from']}",
        to=f"whatsapp:{to_phone}",
    )
    return msg.sid


async def send_fraud_alert(matched_phrases: list, transcript: str) -> None:
    """
    Fire-and-forget WhatsApp alert. Called from the WebSocket handler when
    consecutive_high >= 2. Никогда не бросает наружу — иначе оборвётся стрим.
    """
    cfg = _config()
    if not cfg["to"]:
        print(f"[alert] WhatsApp not configured (no GUARD_CALL_FAMILY_PHONE). "
              f"Would send: phrases={matched_phrases} transcript='{transcript[:80]}'")
        return

    loop = asyncio.get_event_loop()
    try:
        sid = await loop.run_in_executor(
            None, _send_sync, cfg["name"], cfg["to"], matched_phrases
        )
        print(f"[alert] ✅ WhatsApp sent to {cfg['to']} (SID: {sid})")
    except Exception as e:
        print(f"[alert] ❌ WhatsApp send failed: {type(e).__name__}: {e}")
