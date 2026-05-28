import os
from twilio.rest import Client

def send_fraud_alert(victim_name: str, to_phone: str, matched_phrases: list[str]) -> str:
    # Ключи Twilio
    account_sid = "AC7ebd5dce356d00cd185e6dfd1e50aa8b"
    auth_token = "edf034243b08847dbc15ff1cd4749fee"
    from_number = "+14155238886"

    phrases_text = ", ".join(matched_phrases)
    
   
    body = (
        f"🚨 אזהרה! מתקבלת כעת שיחה חשודה אצל {victim_name}. "
        f"זוהו סימני הונאה: {phrases_text}. "
        f"התקשרו אליו/אליה מיד!"
    )

    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body=body,
        from_=f"whatsapp:{from_number}",
        to=f"whatsapp:{to_phone}",
    )

    return message.sid


if __name__ == "__main__":
    MY_NUMBER = "+972537158722" 
    
    print("Sending test fraud alert...")
    
    try:
        sid = send_fraud_alert("אימא", MY_NUMBER, ["חשבון בטוח", "קוד אימות"])
        print(f"✅ Success! Message sent. SID: {sid}")
    except Exception as e:
        print(f"❌ Error occurred: {e}")
