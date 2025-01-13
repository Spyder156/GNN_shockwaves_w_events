import csv
from datetime import datetime

def add_event():
    event_file = "data/events.csv"

    # Check if the file exists; if not, create it and write the header
    try:
        with open(event_file, "x") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "event_type", "event_strength", "affected_node"])
    except FileExistsError:
        pass  # File already exists

    # Prompt for event details
    date = input("Enter event date (YYYY-MM-DD): ")
    try:
        datetime.strptime(date, "%Y-%m-%d")  # Validate date format
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD.")
        return

    event_type = input("Enter event type (e.g., Earnings, News, Regulation): ")
    event_strength = input("Enter event strength (numeric value): ")
    try:
        event_strength = float(event_strength)
    except ValueError:
        print("Event strength must be a numeric value.")
        return

    affected_node = input("Enter affected node (company ticker, or ALL for market-wide event): ")

    # Append event to file
    with open(event_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow([date, event_type, event_strength, affected_node])
    print("Event added successfully.")

add_event()
