
import requests
from datetime import datetime

# 徳山競艇 1-2分析システム
VENUE_CODE = "18"

def check_connection():
    url = "https://www.boatrace.jp/"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        print("公式サイトへの接続成功")
        return True
    except requests.RequestException as e:
        print("接続エラー:", e)
        return False

if __name__ == "__main__":
    print("徳山競艇 1-2分析システム")
    print("実行日時:", datetime.now())
    check_connection()
