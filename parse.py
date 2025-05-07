import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import re
from math import floor


json_file = f"data/train_data_1_cleaned.json"

output_file = f"train_1_pops.json"

# time difference in minutes to be considered late
diff = 10


total = f"data/train_data_1.json"

# averaged the number of trips from the start of the dataset to the end
# manually set the start and end times
def average_trips():
    with open(total, 'r') as f:
        data = json.load(f)

    start_time = "2025-04-26T12:46:21.562829"
    end_time = "2025-05-06T14:18:00.301674"

    # Parse the timestamps into datetime objects
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)

    # Calculate the difference
    delta = end_dt - start_dt

    delta_seconds = delta.total_seconds()

    delta_minutes = delta_seconds / 60

    # 10 minute intervals
    delta_interval = delta_minutes / 10

    trips = data["trains"]
    total_trips = sum(len(v['updates']) for v in trips.values())
   

    # print(len(trips['071150_1..S03R']["updates"]))
    
    average_trips = total_trips / delta_interval

    return average_trips

# returns a list of dictionaries with timestamp as key and number of late vs on time trips as values

def count_pops():


    with open(json_file, 'r') as f:
        data = json.load(f)

    trips = data["trains"]
    pops = {}  # Dictionary to store results
    current_interval = None
    late_trips = 0
    total_trips = 0

    for trip in trips.values():
        updates = trip["updates"]
        for i in range(len(updates) - 1):  # -1 because we need to compare with next update

            update = updates[i]
            next_update = updates[i + 1]
            
            # Check if train is southbound on 1 line and stopped
            if (update["location_status"] == "STOPPED_AT" and 
                update["direction"] == "S"):
                
                timestamp = update["timestamp"]
                next_timestamp = round_to_next_10_min(timestamp)
                
                # If we're starting a new interval, save the previous interval's data
                if current_interval and next_timestamp != current_interval:
                    #on_time_trips = floor(average_trips_total) - late_trips

                    if current_interval not in pops:
                        pops[current_interval] = late_trips
                    else:
                        pops[current_interval] += late_trips

                    late_trips = 0
                
                current_interval = next_timestamp
                
                # Get expected arrival time
                expected_arrival = eta(update["stop_time_updates"])
                try: 
                    if expected_arrival:
                        # Convert HH:MM:SS to datetime using the current date from timestamp
                        current_dt = datetime.fromisoformat(timestamp)
                        expected_time = datetime.strptime(expected_arrival, "%H:%M:%S").time()
                        expected_dt = datetime.combine(current_dt.date(), expected_time)
                        
                        # Get actual arrival time from next update
                        actual_arrival = next_update["last_position_update"]
                        actual_dt = datetime.fromisoformat(actual_arrival)
                        
                        # Check if train is late (more than 10 minutes difference)
                        time_diff = (actual_dt - expected_dt).total_seconds() / 60
                        #print(time_diff)
                        if time_diff > diff:
                            late_trips += 1
                        total_trips += 1
                except:
                    print("Error in expected_arrival")
                    continue
    
    # Save the last interval's data
    if current_interval:
        # on_time_trips = floor(average_trips_total) - late_trips
        pops[current_interval] += late_trips


    # sort pops by chronological order
    sorted_pops = sorted(pops.items())
    
    # Save results to JSON file
    with open(output_file, "w") as f:
        json.dump(sorted_pops, f, indent=4)
    
    return pops

def eta(s):
    # Find the first dictionary entry by finding the first { and }
    start = s.find('{')
    end = s.find('}')
    if start == -1 or end == -1:
        return None
        
    # Extract the first dictionary entry
    entry = s[start:end+1]
    
    # Find the Arr value
    arr_start = entry.find('Arr:')
    if arr_start == -1:
        return None
        
    # Move past 'Arr:' and any whitespace
    arr_start = entry.find(':', arr_start) + 1
    while arr_start < len(entry) and entry[arr_start].isspace():
        arr_start += 1
        
    # Find the end of the value (either a comma or closing brace)
    arr_end = entry.find(',', arr_start)
    if arr_end == -1:
        arr_end = entry.find('}', arr_start)
    
    # Extract and return the Arr value
    return entry[arr_start:arr_end].strip()




def round_to_next_10_min(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    
    # Calculate minutes to next 10-minute mark
    minutes_to_add = 10 - (dt.minute % 10)
    if minutes_to_add == 10 and dt.second == 0 and dt.microsecond == 0:
        # Already aligned to a 10-minute mark
        rounded = dt
    else:
        # Add the required delta to reach the next 10-minute mark
        rounded = dt.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)
    
    return rounded.isoformat()


#print(average_trips())













    

    


