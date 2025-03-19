from nyct_gtfs import NYCTFeed
from nyct_gtfs.compiled_gtfs.nyct_subway_pb2 import NyctFeedHeader
from datetime import datetime
import time

if __name__ == '__main__':


    seen_trains_local = {}
    seen_trains_express = {}

    # with open("seen_trains_local.csv", "r") as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         seen_trains_local[line] = 1
    #
    # with open("seen_trains_express.csv", "r") as f:
    #     lines = f.readlines()[1:]
    #     for line in lines:
    #         seen_trains_express[line] = 1


    while True:
        feed = NYCTFeed("1")

        trains = feed.filter_trips(line_id=["1", "2", "3"], underway=True)


        # 1 trains are always local trains
        # 2 and 3 trains can be express trains
        for train in trains:

            # print(train.location_status)


            # determine if train is local or express
            # 1 - southbound local
            # 2 - southbound express
            # 3 - northbound express
            # 4 - northbound local
            scheduled_type_train = train.stop_time_updates[0].scheduled_track
            actual_type_train = train.stop_time_updates[0].actual_track

            line = train.route_id
            stop = train.location
            toa = train.last_position_update
            is_delayed = train.has_delay_alert
            direction = train.direction
            stop_id = train.current_stop_sequence_index
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

            # string to record into csv files

            # trip_id,line,direction,nyc_train_id,shape_id,unique_id,train_origin,train_assigned,departure_time,scheduled_type_train,location_status,stop,current_stop_index,headsign_text,underway,is_delayed,TOA
            record_str = (
                    str(trip_id) + ',' + str(line) + ',' + str(direction) + "," + str(nyc_train_id) + ',' + str(
                shape) + "," + str(unique_id) + ',' + str(train_origin) + ',' + str(train_assigned) +
                    ',' + str(departure_time) + ',' + str(scheduled_type_train) + "," + str(
                location_status) + ',' + str(stop) + "," + str(
                stop_id) + "," + str(headsign_text) + "," + str(underway) + "," + str(is_delayed) + "," + str(
                toa) + "\n")

            id_str = (
                    str(trip_id) + ',' + str(line) + ',' + str(direction) + "," + str(nyc_train_id) + ',' + str(
                shape) + "," + str(unique_id) + ',' + str(train_origin) + ',' + str(train_assigned) +
                    ',' + str(departure_time) + ',' + str(scheduled_type_train) + "," + str(
                location_status) + ',' + str(stop) + "," + str(
                stop_id) + "," + str(headsign_text) + "," + str(underway) + "," + str(is_delayed) + "\n")


            if (scheduled_type_train != actual_type_train) or seen_trains_express.get(id_str) == 1 or seen_trains_local.get(id_str) == 1 : # if scheduled_type_train differs from actual_type_train then don't record into dataset
                continue

            elif scheduled_type_train == "1" or scheduled_type_train == "4": # for local trains
                local_train_path = "local_train_updatedv2.csv"
                with open(local_train_path, "a") as f:
                    f.write(record_str)
                    f.close()

                seen_trains_local[id_str] = 1

                seen_trains_local_path = "seen_trains_local.csv"
                with open(seen_trains_local_path, "a") as f:
                    f.write(id_str)
                    f.close()

                print(record_str)

            elif scheduled_type_train == "2" or scheduled_type_train == "3":

                with open("express_train_updatedv2.csv", "a") as f:
                    f.write(record_str)
                    f.close()

                seen_trains_express[id_str] = 1

                seen_trains_express_path = "seen_trains_express.csv"
                with open(seen_trains_express_path, "a") as f:
                    f.write(id_str)
                    f.close()

                print(record_str)

        # just so program doesn't kill my computer
        time.sleep(1)