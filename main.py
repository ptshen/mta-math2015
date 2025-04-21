from nyct_gtfs import NYCTFeed
from nyct_gtfs.compiled_gtfs.nyct_subway_pb2 import NyctFeedHeader
from datetime import datetime
import time
import json
import os

def load_json_file(file_path, default=None):
    """Load JSON from a file with error handling"""
    if default is None:
        default = {}
    
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return default
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading {file_path}, starting fresh")
        return default

def save_json_file(file_path, data):
    """Save JSON to a file with proper encoding"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def is_duplicate_state(current_state, last_state):
    """Check if the current state is the same as the last recorded state"""
    if not last_state:
        return False
    
    return (current_state["stop"] == last_state["stop"] and 
            current_state["location_status"] == last_state["location_status"])

if __name__ == '__main__':
    seen_trains = {}  # Track which trains we've seen
    lines = ["A", "B", "G", "J", "N", "L", "1"]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    while True:
        for line in lines: 
            try:
                feed = NYCTFeed(line)
                trains = feed.filter_trips(underway=True)
                print(f"Found {len(trains)} trains on line {line}")

                # Path to the JSON file for this line
                line_file = f"data/train_data_{line}.json"
                
                # Load existing line data
                line_data = load_json_file(line_file, {
                    "line": line,
                    "trains": {}
                })

                for train in trains:
                    try:
                        line = train.route_id
                        stop = train.location
                        lpu = train.last_position_update
                        is_delayed = train.has_delay_alert
                        direction = train.direction
                        stop_id = train.current_stop_sequence_index
                        stop_time_updates = train.stop_time_updates

                        location_status = train.location_status
                        train_assigned = train.train_assigned
                        headsign_text = train.headsign_text
                        underway = train.underway

                        trip_id = train.trip_id
                        train_origin = train.start_date
                        nyc_train_id = train.nyc_train_id

                        shape = train.shape_id
                        departure_time = train.departure_time

                        # Create current state record
                        current_state = {
                            "trip_id": str(trip_id),
                            "line": str(line),
                            "direction": str(direction),
                            "nyc_train_id": str(nyc_train_id),
                            "shape_id": str(shape),
                            "train_origin": str(train_origin),
                            "train_assigned": str(train_assigned),
                            "departure_time": str(departure_time),
                            "location_status": str(location_status),
                            "stop": str(stop),
                            "stop_id": str(stop_id),
                            "headsign_text": str(headsign_text),
                            "underway": str(underway),
                            "is_delayed": str(is_delayed),
                            "last_position_update": str(lpu),
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        # Initialize train data if it doesn't exist
                        if trip_id not in line_data["trains"]:
                            line_data["trains"][trip_id] = {
                                "trip_id": trip_id,
                                "updates": []
                            }
                        
                        # Get the last state for this train
                        train_updates = line_data["trains"][trip_id]["updates"]
                        last_state = train_updates[-1] if train_updates else None
                        
                        # Check if this is a new state
                        if not is_duplicate_state(current_state, last_state):
                            # Add the new state to the updates list
                            train_updates.append(current_state)
                            
                            print(f"Added new state for trip {trip_id} at {stop} with status {location_status}")
                        
                        # Mark as seen
                        seen_trains[trip_id] = 1
                            
                    except Exception as e:
                        print(f"Error processing train: {e}")
                        continue
                
                # Save the updated line data
                save_json_file(line_file, line_data)
                
            except Exception as e:
                print(f"Error fetching feed for line {line}: {e}")
                continue
                
            # Delay between processing each line
            time.sleep(1)
        
        # Delay between full cycles
        print("Completed cycle, waiting 5 seconds...")
        time.sleep(5)