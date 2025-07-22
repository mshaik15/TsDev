from collections import Counter

def infer_time_interval(df):
    deltas = [] # Store time differences between consecutive timestamps
    
    for i in range(1, len(df)):
        delta = df.timestamp[i] - df.timestamp[i - 1]
        deltas.append(delta)

    # Find the mode of the time differences
    delta_counts = Counter(deltas)
    t_rec, _ = delta_counts.most_common(1)[0]

    response = input(f"Time interval inferred: {t_rec}, continue with this value? (y/n).")
    if response.lower() == 'y' or 'yes':
        print(f"Using time interval: {t_rec}")
        return t_rec
    
    else:
        t_rec = input("Please enter the time interval in seconds:")

        try:
            t_rec = int(t_rec)
            print(f"Using user-defined time interval: {t_rec}")
            return t_rec
        
        except ValueError:
            print("Invalid input. Using default time interval of 1 second.")
            return 1