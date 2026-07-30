from datetime import datetime, timezone, timedelta

WIB = timezone(timedelta(hours=7))

def get_time_now_wib():
    return datetime.now(WIB)

