
import requests

BASE_URL = "http://localhost:8000"

def test_activity():
    print("\n--- Testing Activity Report ---")
    params = {
        "start_date": "2026-01-01",
        "end_date": "2026-02-01",
        "mr_id": "MR_W1_2"
    }
    try:
        r = requests.get(f"{BASE_URL}/reports/activity", params=params)
        if r.status_code == 200:
            data = r.json()
            print(f"Success! Total Activities: {data.get('total_activities')}")
            print(f"Rows: {len(data.get('data', []))}")
        else:
            print(f"Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_compliance():
    print("\n--- Testing Compliance Report ---")
    params = {
        "month": 1,
        "year": 2026,
        "mr_id": "MR_W1_2"
    }
    try:
        r = requests.get(f"{BASE_URL}/reports/compliance", params=params)
        if r.status_code == 200:
            data = r.json()
            print(f"Success! Rows: {len(data)}")
            if len(data) > 0:
                print(f"Sample: {data[0]}")
        else:
            print(f"Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_activity()
    test_compliance()
