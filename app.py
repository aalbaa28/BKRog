import streamlit as st
import pandas as pd
import json
import os

# Function to load all JSON files from a folder and combine them into a DataFrame
def load_json_data(folder):
    combined_data = []
    try:
        # List all JSON files in the folder
        json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
        
        for file in json_files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                combined_data.append(json_data)
        
        st.success(f"Data loaded successfully from {folder}")
        return combined_data
    except Exception as e:
        st.error(f"Error loading JSON files: {e}")
        return None

def extract_player_data(participant):
    challenges = participant['challenges']
    champion_id = participant['championId']
    champion_image_url = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/{champion_id}.png"
    
    data = {
        'championImage': champion_image_url,  # URL of the champion's image
        'championName': participant['championName'],
        'win': participant['win'],
        'kda': round(challenges['kda'], 2),
        'deaths': participant['deaths'],
        'goldPerMinute': round(challenges['goldPerMinute'], 2),
        'damagePerMinute': round(challenges['damagePerMinute'], 2),
        'teamDamagePercentage': round(challenges['teamDamagePercentage'], 2),
        'side': 'blue' if participant['teamId'] == 100 else 'red',  # Determine side based on teamId
    }
    return data

# Function to convert match data into a DataFrame
def get_matchup_data(json_data):
    matchups = []
    
    # List of players we are interested in
    players_of_interest = ["BKR Szygenda", "BKR Rhilech", "BKR OMON", "WD BOOSHI", "BKR Doss"]
    
    # Iterate through all participants in the game
    for participant in json_data['participants']:
        if participant['riotIdGameName'] in players_of_interest:
            # Extract player data
            player_data = extract_player_data(participant)
            
            # Determine player position
            if participant['riotIdGameName'] == "BKR Szygenda":
                player_data['Position'] = 'Top'
            elif participant['riotIdGameName'] == "BKR Rhilech":
                player_data['Position'] = 'Jgl'
            elif participant['riotIdGameName'] == "BKR OMON":
                player_data['Position'] = 'Mid'
            elif participant['riotIdGameName'] == "WD BOOSHI":
                player_data['Position'] = 'Adc'
            elif participant['riotIdGameName'] == "BKR Doss":
                player_data['Position'] = 'Supp'
            
            # Find the opponent in the same lane
            team_id = participant['teamId']
            participant_index = json_data['participants'].index(participant)  # Get current participant's index
            
            if team_id == 100:
                # If teamId is 100, the opponent will be 5 positions ahead
                opponent_index = participant_index + 5
            elif team_id == 200:
                # If teamId is 200, the opponent will be 5 positions before
                opponent_index = participant_index - 5

            opponent = json_data['participants'][opponent_index]
            
            # Add enemy champion to the data dictionary
            if opponent:
                player_data['EnemyChampion'] = opponent['championName']
                player_data['EnemyChampionImage'] = f"https://raw.communitydragon.org/latest/plugins/rcp-be-lol-game-data/global/default/v1/champion-icons/{opponent['championId']}.png"
            else:
                player_data['EnemyChampion'] = "Unknown"
                player_data['EnemyChampionImage'] = None
            
            # Add additional game information
            player_data['Date'] = pd.to_datetime(json_data['gameCreation'], unit='ms')
            player_data['gameName'] = json_data['gameName']
            player_data['gameVersion'] = json_data['gameVersion']
            
            matchups.append(player_data)
    
    df = pd.DataFrame(matchups)
    return df

# Function to calculate the average metrics per champion
def calculate_average_by_champion(df, position=None):
    if position:
        df = df[df['Position'] == position]
    
    # Group by championName and calculate the mean for numeric columns
    avg_df = df.groupby('championName').agg({
        'kda': 'mean',
        'deaths': 'mean',
        'goldPerMinute': 'mean',
        'damagePerMinute': 'mean',
        'teamDamagePercentage': 'mean',
        'side': 'count',  # Count number of appearances
        'championImage': 'first',  # Take the first image for the champion (it should be the same for each)
        'win': 'sum'  # Sum of wins to calculate WR
    }).reset_index()
    
    # Calculate Win Rate (WR) as (wins / total games) * 100
    avg_df['winrate'] = (avg_df['win'] / avg_df['side']) * 100
    
    # Sort by total games played (side count) and then by KDA for better display
    avg_df = avg_df.sort_values(by='side', ascending=False)
    
    return avg_df



# Load data from the folder
json_folder = json_folder = "March 18"  # Ahora buscará dentro del repo en Streamlit Cloud
  # Change this to your folder path
json_data = load_json_data(json_folder)

if json_data:
    # Combine all JSON data into a single DataFrame
    combined_df = pd.DataFrame()
    
    for data in json_data:
        # Get matchup data for this JSON
        df = get_matchup_data(data)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        combined_df = combined_df.sort_values(by='Date', ascending=False)

    # Filter by side (blue or red)
    side_filter = st.sidebar.selectbox("Filter by side", ['All', 'blue', 'red'])

    # Apply side filter
    if side_filter != 'All':
        combined_df = combined_df[combined_df['side'] == side_filter]

    # Get the list of unique champions
    champion_list = combined_df['championName'].unique().tolist()
    champion_list.sort()  # Sort alphabetically
    champion_list.insert(0, "All")  # Add "All" option to disable the filter

    # Filter by champion
    champion_filter = st.sidebar.selectbox("Filter by champion", champion_list)

    # Apply champion filter
    if champion_filter != "All":
        combined_df = combined_df[combined_df['championName'] == champion_filter]

        # Add the date filters in the sidebar
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2025-03-01'))  # Default start date
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2025-03-31'))  # Default end date

    # Convertir las fechas seleccionadas a formato datetime (asegurarse de incluir la hora en end_date)
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date).replace(hour=23, minute=59, second=59)  # Aseguramos que sea hasta el final del día

    # Filtrar el DataFrame por el rango de fechas
    if 'Date' in combined_df.columns:
        combined_df = combined_df[(combined_df['Date'] >= start_date) & (combined_df['Date'] <= end_date)]


    # Create tabs for each position
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Top", "Jgl", "Mid", "Adc", "Supp", "By Champion"])

    def display_table_with_images(df, position):
        st.header(position)
        pos_data = df[df['Position'] == position]
        
        if not pos_data.empty:
            for index, row in pos_data.iterrows():
                col1, col2, col3 = st.columns([1, 1, 4])  # Add a column for the enemy champion
                with col1:
                    # Ensure 'championImage' exists before attempting to access it
                    champion_image = row.get('championImage', None)
                    if champion_image:
                        st.image(champion_image, width=50, caption=row['championName'])  # Player's champion
                    else:
                        st.write("No image available")
                
                with col2:
                    if row['EnemyChampionImage']:
                        st.image(row['EnemyChampionImage'], width=50, caption=row['EnemyChampion'])  # Enemy champion
                    else:
                        st.write("Unknown")
                
                with col3:
                    st.write(f"**{row['championName']} vs {row['EnemyChampion']}**")
                    st.write(f"Date: {row['Date'].strftime('%Y-%m-%d')}")
                    win_color = "green" if row["win"] else "red"
                    st.markdown(f"<span style='color:{win_color}; font-weight:bold;'>{'Win' if row['win'] else 'Loss'}</span>", unsafe_allow_html=True)

                    st.write(f"KDA: {row['kda']} | Deaths: {row['deaths']}")
                    st.write(f"Gold per minute: {row['goldPerMinute']} | Damage per minute: {row['damagePerMinute']}")
                    st.write(f"Team damage percentage: {row['teamDamagePercentage']*100:.2f}% | Side: {row['side']}")
                    st.write("---")
        else:
            st.write(f"No data available for {position}.")

    with tab1:
        display_table_with_images(combined_df, 'Top')

    with tab2:
        display_table_with_images(combined_df, 'Jgl')

    with tab3:
        display_table_with_images(combined_df, 'Mid')

    with tab4:
        display_table_with_images(combined_df, 'Adc')

    with tab5:
        display_table_with_images(combined_df, 'Supp')

    # Calculate average metrics by champion and display in the new tab
    with tab6:
        st.header("Average Metrics by Champion")
        
        # Add a position filter for this tab
        position_filter = st.selectbox("Filter by position", ['All', 'Top', 'Jgl', 'Mid', 'Adc', 'Supp'])
        
        # Calculate the average metrics, considering the selected position
        avg_champion_df = calculate_average_by_champion(combined_df, position_filter if position_filter != 'All' else None)
        
        # Display the champion summary with image, count, and average stats
        for index, row in avg_champion_df.iterrows():
            col1, col2, col3 = st.columns([1, 4, 3])
            
            with col1:
                # Ensure 'championImage' exists before attempting to access it
                champion_image = row.get('championImage', None)
                if champion_image:
                    st.image(champion_image, width=50, caption=row['championName'])
                else:
                    st.write("No image available")
            
            with col2:
                st.subheader(f"{row['championName']}")
                st.write(f"**Games Played**: {row['side']}")
                winrate_color = "green" if row['winrate'] >= 50 else "red"
                st.markdown(f"<p style='color:{winrate_color};'><b>Winrate:</b> {row['winrate']:.2f}%</p>", unsafe_allow_html=True)

                st.write(f"**Average KDA**: {row['kda']:.2f}")
                st.write(f"**Average Deaths**: {row['deaths']:.2f}")
                st.write(f"**Gold per Minute**: {row['goldPerMinute']:.2f}")
                st.write(f"**Damage per Minute**: {row['damagePerMinute']:.2f}")
                st.write(f"**Team Damage Percentage**: {row['teamDamagePercentage']*100:.2f}%")
            
            st.write("---")

    # Save the filtered results as a CSV
    if st.button('Save CSV'):
        csv_file = 'filtered_matchups.csv'
        combined_df.to_csv(csv_file, index=False)
        st.success(f"CSV saved as {csv_file}")
