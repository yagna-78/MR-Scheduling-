
import pandas as pd

def test_enrichment():
    # Mock data
    data_master = [
        {'activity_id': '1', 'mr_id': 'A', 'status': 'Planned', 'travel_duration_min': 10, 'distance_km': 5, 'suggested_talking_points': 'Talk about X'},
        {'activity_id': '2', 'mr_id': 'A', 'status': 'Planned', 'travel_duration_min': 20, 'distance_km': 10, 'suggested_talking_points': 'Talk about Y'},
        {'activity_id': '3', 'mr_id': 'A', 'status': 'Planned', 'travel_duration_min': 30, 'distance_km': 15, 'suggested_talking_points': 'Talk about Z'},
    ]
    
    # Activity 2 is completed, so it is in activities table. It usually lacks the travel/static fields.
    data_activities = [
        {'activity_id': '2', 'mr_id': 'A', 'status': 'Done', 'visit_count': 1}, # Missing travel info
    ]
    
    print("Original Activities Data:", data_activities)
    
    # --- LOGIC COPIED FROM schedule.py ---
    if data_activities and data_master:
        master_map = {row['activity_id']: row for row in data_master}
        enrich_keys = ['travel_duration_min', 'distance_km', 'suggested_talking_points']
        
        for row in data_activities:
            if 'activity_id' in row and row['activity_id'] in master_map:
                m_row = master_map[row['activity_id']]
                for key in enrich_keys:
                    # Only fill if missing or empty in activity row
                    if key not in row or row[key] is None or row[key] == "":
                         if key in m_row:
                             row[key] = m_row[key]
    # -------------------------------------
    
    print("Enriched Activities Data:", data_activities)
    
    # Verify
    act_2 = data_activities[0]
    if act_2.get('travel_duration_min') == 20:
        print("SUCCESS: Travel duration added.")
    else:
        print("FAILURE: Travel duration missing.")

    if act_2.get('distance_km') == 10:
        print("SUCCESS: Distance added.")
    else:
        print("FAILURE: Distance missing.")

    if act_2.get('status') == 'Done':
        print("SUCCESS: Status preserved as Done.")
    else:
        print("FAILURE: Status overwritten.")

if __name__ == "__main__":
    test_enrichment()
