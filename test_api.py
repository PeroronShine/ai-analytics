import requests
import json
import sys

def test_gen_api(token: str) -> bool:
    """Тестирует подключение к gen-api.ru Qwen3.6"""
    
    url = "https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Простой запрос для проверки
    payload = {
        "messages": [
            {"role": "system", "content": "Ты помощник. Отвечай кратко."},
            {"role": "user", "content": "Напиши код Python: print('API работает!')"}
        ],
        "is_sync": True,          # Синхронный режим (ждём ответ сразу)
        "temperature": 0.3,
        "max_tokens": 150
    }
    
    print("🔍 Отправляю запрос к API...")
    print(f" Endpoint: {url}")
    print(f" Токен: {token[:6]}...{token[-4:]}")
    print("-" * 40)
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        print(f"📥 Статус HTTP: {response.status_code}")
        print(f"📄 Заголовки ответа: {dict(response.headers)}")
        print("-" * 40)
        
        # Успешный ответ
        if response.status_code == 200:
            data = response.json()
            print("✅ УСПЕХ! Ответ получен.")
            print("\n📦 Полный JSON ответа:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Пробуем извлечь текст разными способами (зависит от версии API)
            text = (data.get("output") or 
                   data.get("result") or 
                   data.get("choices", [{}])[0].get("message", {}).get("content", ""))
            
            if text:
                print("\n💬 Извлечённый текст:")
                print(text)
            else:
                print("\n️ Текст не найден в стандартных полях. Проверьте структуру выше.")
            return True
            
        # Ошибки
        else:
            print(f"❌ ОШИБКА API: {response.status_code}")
            print("📜 Тело ответа:")
            try:
                print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            except:
                print(response.text)
            return False
            
    except requests.exceptions.Timeout:
        print("⏱ ТАЙМАУТ: Сервер не ответил за 30 сек")
        return False
    except requests.exceptions.ConnectionError:
        print(" ОШИБКА СОЕДИНЕНИЯ: Нет интернета или блокировка")
        return False
    except requests.exceptions.RequestException as e:
        print(f"💥 ОШИБКА ЗАПРОСА: {e}")
        return False
    except Exception as e:
        print(f"💥 НЕОЖИДАННАЯ ОШИБКА: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print(" Тест подключения к gen-api.ru (Qwen3.6)")
    print("Для выхода нажмите Ctrl+C\n")
    
    token = input("🔑 Введите токен: ").strip()
    if not token:
        print("❌ Токен пустой. Завершение.")
        sys.exit(1)
        
    print("\n" + "="*50)
    success = test_gen_api(token)
    print("="*50)
    
    if success:
        print("✅ API работает корректно! Проблема в UI/Streamlit.")
    else:
        print("❌ API вернул ошибку. Проверьте токен, баланс или лимиты.")
