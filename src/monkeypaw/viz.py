import matplotlib.pyplot as plt

def plot_cursor_trajectory(df_gpb_session_trial, max_abs_val=None):
    
    """ 
    `plot_cursor_trajectory` calculates target direction and plots the cursor trajectory,
    along with event markers. It takes `df_gpb_session_trial` as its only argument
    
    Args:
    df_gpb_session_trial: pd dataframe grouped by session and trial for animal x
    max_abs_val: symmetric limits based on the maximum absolute value, centered at 0
    """
    
    if df_gpb_session_trial.empty:
        print("DataFrame is empty, cannot plot trajectory.")
        return

    # Calculate unique target directions
    unique_target_dirs_rad = df_gpb_session_trial['target_dir'].unique()
    unique_target_dirs_deg = [np.degrees(direction) for direction in unique_target_dirs_rad]

    # Format target direction for title and labels, handling NaN
    target_dir_title = "N/A"
    if unique_target_dirs_deg and not pd.isna(unique_target_dirs_deg[0]):
        target_dir_title = f"{int(unique_target_dirs_deg[0])}°"

    # Print statement for target direction (original code had this before the if/else for cursor_pos_x.isnull().all())
    if unique_target_dirs_deg:
        print('Target direction for session ',df_gpb_session_trial['session'].unique()[0],', trial ', df_gpb_session_trial['trial_id'].unique()[0],': ', unique_target_dirs_deg[0], ' degrees')


    # Check if cursor position data is all NaN
    if df_gpb_session_trial['cursor_pos_x'].isnull().all() or df_gpb_session_trial['cursor_pos_y'].isnull().all():
        print(f"Warning: No valid cursor position data (cursor_pos_x or cursor_pos_y are all NaN) for Animal {df_gpb_session_trial['animal'].iloc[0]}, Session {df_gpb_session_trial['session'].iloc[0]}, Trial {df_gpb_session_trial['trial_id'].iloc[0]}. Plotting axes with default limits.")

        plt.figure(figsize=(8, 8))
        plt.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=0)
        plt.axvline(0, color='black', linewidth=1.5, linestyle='-', zorder=0)

        # default limits when no cursor data
        default_limit = 10.0 # default for cursor positions
        plt.xlim(-default_limit, default_limit)
        plt.ylim(-default_limit, default_limit)

        plt.xlabel('Cursor Position X (cm)')
        plt.ylabel('Cursor Position Y (cm)')

        result_dic = {'R':'Rewarded','A':'Aborted','F':'Failed','I':'Incomplete'}
        # Ensure 'result' exists and is not empty
        if not df_gpb_session_trial['result'].empty:
            result_str = result_dic.get(df_gpb_session_trial['result'].iloc[0], 'Unknown')
        else:
            result_str = 'N/A'

        # line for target direction
        if unique_target_dirs_deg and not pd.isna(unique_target_dirs_deg[0]):
            target_angle_rad = np.radians(unique_target_dirs_deg[0])
            line_length = 15
            target_line_x = [0, line_length * np.cos(target_angle_rad)]
            target_line_y = [0, line_length * np.sin(target_angle_rad)]
            plt.plot(target_line_x, target_line_y, linestyle='--', color='yellow', linewidth=2, zorder=1, label=f'Target Direction ({target_dir_title})')

        plt.title(f"Cursor Trajectory for Dataset {df_gpb_session_trial['datasetID'].iloc[0]}: Animal {df_gpb_session_trial['animal'].iloc[0]}, Session {df_gpb_session_trial['session'].iloc[0]}, Trial {df_gpb_session_trial['trial_id'].iloc[0]} - Target {target_dir_title}, result {result_str}")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        return


    # Get trial start time
    trial_start_time = df_gpb_session_trial['time_s'].iloc[0]

    # Draw darker axis lines at 0,0
    plt.figure(figsize=(8, 8))
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=0)
    plt.axvline(0, color='black', linewidth=1.5, linestyle='-', zorder=0)

    plt.plot(df_gpb_session_trial['cursor_pos_x'], df_gpb_session_trial['cursor_pos_y'], marker='.', linestyle='-', label='Cursor Trajectory')
    plt.plot(df_gpb_session_trial['cursor_pos_x'].iloc[0], df_gpb_session_trial['cursor_pos_y'].iloc[0], marker='*', color='red', markersize=15, label='Start Point')

    # Mark EventTarget_Onset points
    target_onset_rows = df_gpb_session_trial[df_gpb_session_trial['EventTarget_Onset'] == True]
    target_onset_x = target_onset_rows['cursor_pos_x']
    target_onset_y = target_onset_rows['cursor_pos_y']
    if not target_onset_rows.empty:
        relative_target_onset_time = target_onset_rows['time_s'].iloc[0] - trial_start_time
        plt.plot(target_onset_x, target_onset_y, 'o', color='green', markersize=10, label=f'Target Onset ({relative_target_onset_time:.2f} s)')
    else:
        plt.plot([], [], 'o', color='green', markersize=10, label='Target Onset (N/A)') # Plot empty if no events

    # Mark EventGo_cue points
    go_cue_rows = df_gpb_session_trial[df_gpb_session_trial['EventGo_cue'] == True]
    go_cue_x = go_cue_rows['cursor_pos_x']
    go_cue_y = go_cue_rows['cursor_pos_y']
    if not go_cue_rows.empty:
        relative_go_cue_time = go_cue_rows['time_s'].iloc[0] - trial_start_time
        plt.plot(go_cue_x, go_cue_y, 'x', color='purple', markersize=10, label=f'Go Cue ({relative_go_cue_time:.2f} s)')
    else:
        plt.plot([], [], 'x', color='purple', markersize=10, label='Go Cue (N/A)') # Plot empty if no events

    # Mark the endpoint
    endpoint_row = df_gpb_session_trial.iloc[-1]
    endpoint_x = endpoint_row['cursor_pos_x']
    endpoint_y = endpoint_row['cursor_pos_y']
    relative_endpoint_time = endpoint_row['time_s'] - trial_start_time
    plt.plot(endpoint_x, endpoint_y, 's', color='blue', markersize=10, label=f'Endpoint ({relative_endpoint_time:.2f} s)')

    # Calculate symmetric ranges for x and y axes centered at 0
    x_min_data = df_gpb_session_trial['cursor_pos_x'].min()
    x_max_data = df_gpb_session_trial['cursor_pos_x'].max()
    y_min_data = df_gpb_session_trial['cursor_pos_y'].min()
    y_max_data = df_gpb_session_trial['cursor_pos_y'].max()

    # Determine the maximum absolute value among all min/max data points
    if max_abs_val is None:
        max_abs_val = max(abs(x_min_data), abs(x_max_data), abs(y_min_data), abs(y_max_data))+0.5

    # Set new symmetric limits based on the maximum absolute value, centered at 0
    plt.xlim(-max_abs_val, max_abs_val)
    plt.ylim(-max_abs_val, max_abs_val)

    # Add line for target direction
    if unique_target_dirs_deg and not pd.isna(unique_target_dirs_deg[0]):
        target_angle_rad = np.radians(unique_target_dirs_deg[0])
        line_length = max_abs_val * 1.5
        target_line_x = [0, line_length * np.cos(target_angle_rad)]
        target_line_y = [0, line_length * np.sin(target_angle_rad)]
        plt.plot(target_line_x, target_line_y, linestyle='--', color='yellow', linewidth=2, zorder=1, label=f'Target Direction ({target_dir_title})')
    else:
        plt.plot([], [], linestyle='--', color='yellow', linewidth=2, zorder=1, label='Target Direction (N/A)')


    result_dic = {'R':'Rewarded','A':'Aborted','F':'Failed','I':'Incomplete'}
    result_str = result_dic[df_gpb_session_trial['result'].iloc[0]]

    plt.xlabel('Cursor Position X (cm)')
    plt.ylabel('Cursor Position Y (cm)')
    plt.title(f"Cursor Trajectory for Dataset {df_gpb_session_trial['datasetID'].iloc[0]}: Animal {df_gpb_session_trial['animal'].iloc[0]}, Session {df_gpb_session_trial['session'].iloc[0]}, Trial {df_gpb_session_trial['trial_id'].iloc[0]} - Target {target_dir_title}, result {result_str}")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box') # Equal scaling for x and y axes
    plt.legend()
    plt.show()


def plot_events_over_time(df_gpb_session_trial):

    """
    Displays the occurrences of events ('EventTarget_Onset', 'EventGo_cue') as vertical lines
    on a time-series plot for a given animal, session, and trial. The time is relative to
    the start of the trial.

    Args:
        df_gpb_session_trial (pd.DataFrame): DataFrame containing trial data for a specific
                                           animal, session, and trial, including 'time_s',
                                           'EventTarget_Onset', 'EventGo_cue', 'animal',
                                           'session', 'trial_id', 'datasetID', 'result', and
                                           'target_dir' columns.
    """

    if df_gpb_session_trial.empty:
        print("DataFrame is empty, cannot plot events.")
        return

    # Get trial start time for relative timing
    trial_start_time = df_gpb_session_trial['time_s'].iloc[0]

    # Prepare event data
    events = {
        'EventTarget_Onset': 'green',
        'EventGo_cue': 'purple'
    }

    plt.figure(figsize=(12, 4))

    # Add line for the start of the trial
    plt.axvline(0, color='blue', linestyle='-', linewidth=2, label='Trial Start (Relative: 0.00 s)')

    # Plot events as vertical lines
    for event_name, color in events.items():
        event_times = df_gpb_session_trial[df_gpb_session_trial[event_name] == True]['time_s']
        if not event_times.empty:
            for t in event_times:
                relative_time = t - trial_start_time
                plt.axvline(relative_time, color=color, linestyle='--', label=f'{event_name} (Relative: {relative_time:.2f} s)')

    # Add line for the end of the trial
    trial_end_time = df_gpb_session_trial['time_s'].iloc[-1]
    relative_trial_end_time = trial_end_time - trial_start_time
    plt.axvline(relative_trial_end_time, color='red', linestyle='-', linewidth=2, label=f'Trial End (Relative: {relative_trial_end_time:.2f} s)')

    # Set title and labels
    animal = df_gpb_session_trial['animal'].iloc[0]
    session = df_gpb_session_trial['session'].iloc[0]
    trial_id = df_gpb_session_trial['trial_id'].iloc[0]
    dataset_id = df_gpb_session_trial['datasetID'].iloc[0]
    result_dic = {'R':'Rewarded','A':'Aborted','F':'Failed','I':'Incomplete'}
    result_str = result_dic[df_gpb_session_trial['result'].iloc[0]]

    # Handle NaN in target_dir for display
    target_dir_val = df_gpb_session_trial['target_dir'].iloc[0]
    if pd.isna(target_dir_val):
        target_dg = 'N/A'
    else:
        target_dg = int(np.degrees(target_dir_val))

    plt.title(f'Events Over Time for Dataset {dataset_id}: Animal {animal}, Session {session}, Trial {trial_id} - Target {target_dg}°, result {result_str}')
    plt.xlabel('Time Relative to Trial Start (s)')
    plt.ylabel('Event Occurrence')
    plt.yticks([]) # Remove y-axis ticks as they are not needed
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_events_by_brain_region(df):
    """
    Plots events over time for 'M1', 'PMd', 'Area2' and NaN brain regions separately
    from a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing trial data with 'brain_region' column.
    """

    # Plot for M1 brain region
    m1_df = df[df['brain_region'] == 'M1']
    if not m1_df.empty:
        print("\nPlotting events for M1 Brain Region:")
        plot_events_over_time(m1_df)
    else:
        print("\nNo data found for M1 Brain Region.")

    # Plot for PMd brain region
    pmd_df = df[df['brain_region'] == 'PMd']
    if not pmd_df.empty:
        print("\nPlotting events for PMd Brain Region:")
        plot_events_over_time(pmd_df)
    else:
        print("\nNo data found for PMd Brain Region.")

    # Plot for Area2 brain region
    pmd_df = df[df['brain_region'] == 'Area2']
    if not pmd_df.empty:
        print("\nPlotting events for Area2 Brain Region:")
        plot_events_over_time(pmd_df)
    else:
        print("\nNo data found for Area2 Brain Region.")

    # Plot for NaN brain region
    nan_brain_region_df = df[df['brain_region'].isnull()]
    if not nan_brain_region_df.empty:
        print("\nPlotting events for NaN Brain Region:")
        plot_events_over_time(nan_brain_region_df)
    else:
        # Check if 'brain_region' column exists and has non-NaN values
        if 'brain_region' in df.columns and not df['brain_region'].isnull().all():
            print("\nNo NaN brain region data found in the current DataFrame for plotting.")
        else:
            print("\n'brain_region' column is either missing or entirely NaN. No NaN brain region plots to generate.")

    
