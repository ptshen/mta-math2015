from nyct_gtfs import NYCTFeed
from nyct_gtfs.compiled_gtfs.nyct_subway_pb2 import NyctFeedHeader
from datetime import datetime
import time
import json
import os

if __name__ == '__main__':
    seen_trains = {}
    lines = ["A", "B", "G", "J", "N", "L", "1"]
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    while True:
        for line in lines: 
            try:
                feed = NYCTFeed(line)
                trains = feed.filter_trips(underway=True)
                #print(f"Found {len(trains)} trains on line {line}")


                for train in trains:
                    try:
                        line = train.route_id
                        stop = train.location
                        toa = train.last_position_update
                        is_delayed = train.has_delay_alert
                        direction = train.direction
                        stop_id = train.current_stop_sequence_index
                        # stop_time_updates includes predicted time arrival
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

                        unique_id = str(trip_id) + " " + str(train_origin)
                        id_str = (str(trip_id) + ',' + str(line) + ',' + str(direction) + "," + str(nyc_train_id) + ',' + str(shape) + "," + str(unique_id) + ',' + str(train_origin) + ',' + str(train_assigned) +
                    ',' + str(departure_time) + ',' + str(location_status) + ',' + str(stop) + "," + str(stop_id) + "," + str(headsign_text) + "," + str(underway) + "," + str(is_delayed))

                        # Only process if we haven't seen this train before
                        if seen_trains.get(id_str) != 1:
                            record_info = {
                                "trip_id": str(trip_id),
                                "line": str(line),
                                "direction": str(direction),
                                "nyc_train_id": str(nyc_train_id),
                                "shape_id": str(shape),
                                "unique_id": str(unique_id),
                                "train_origin": str(train_origin),
                                "train_assigned": str(train_assigned),
                                "departure_time": str(departure_time),
                                "location_status": str(location_status),
                                "stop": str(stop),
                                "stop_id": str(stop_id),
                                "headsign_text": str(headsign_text),
                                "underway": str(underway),
                                "is_delayed": str(is_delayed),
                                "TOA": str(toa),
                                "stop_time_updates": str(stop_time_updates),
                                "timestamp": datetime.now().isoformat()
                            }
                        
                            # Path to the JSON file for this line
                            local_train_path = f"data/train_data_{line}.json"
                            
                            # Initialize with empty dictionary
                            data = {}
                            
                            # Load existing data if file exists and has content
                            if os.path.exists(local_train_path) and os.path.getsize(local_train_path) > 0:
                                try:
                                    with open(local_train_path, "r") as f:
                                        data = json.load(f)
                                except json.JSONDecodeError:
                                    print(f"Error reading {local_train_path}, starting fresh")
                                    continue
                            
                            # Add the new record
                            data[id_str] = record_info
                            
                            # Write back to the file
                            with open(local_train_path, "w") as f:
                                json.dump(data, f, indent=4)
                            
                            # Mark as seen
                            seen_trains[id_str] = 1
                            
                            # Record to seen_trains.csv
                            with open("seen_trains.csv", "a") as f:
                                f.write(id_str + '\n')
                            
                            #print(f"Added train {line} ({id_str}) at {stop}")
                    except Exception as e:
                        print(f"Error processing train: {e}")
                        continue
            except Exception as e:
                print(f"Error fetching feed for line {line}: {e}")
                continue
                
            # Delay between processing each line
            time.sleep(1)
        
        # Delay between full cycles
        #print("Completed cycle, waiting 5 seconds...")
        time.sleep(5)