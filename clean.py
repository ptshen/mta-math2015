import json

target = f"data/train_data_1.json"
output = f"data/train_data_1_cleaned.json"
def clean_train_data():
    # Read the original JSON file
    with open(target, 'r') as f:
        data = json.load(f)

    # Create a new dictionary to store filtered data
    filtered_data = {"trains": {}}

    # Iterate through each train
    for train_id, train_data in data["trains"].items():
        # Filter updates that match our criteria
        filtered_updates = [
            update for update in train_data["updates"]
            if update["direction"] == "S" and update["location_status"] == "STOPPED_AT"
        ]
        
        # Only keep trains that have matching updates
        if filtered_updates:
            filtered_data["trains"][train_id] = {
                "updates": filtered_updates
            }

    # Save the filtered data to a new JSON file
    with open(output, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    # Print some statistics
    original_trains = len(data["trains"])
    filtered_trains = len(filtered_data["trains"])
    original_updates = sum(len(train["updates"]) for train in data["trains"].values())
    filtered_updates = sum(len(train["updates"]) for train in filtered_data["trains"].values())
    
    print(f"Original trains: {original_trains}")
    print(f"Filtered trains: {filtered_trains}")
    print(f"Original updates: {original_updates}")
    print(f"Filtered updates: {filtered_updates}")

if __name__ == "__main__":
    clean_train_data()
