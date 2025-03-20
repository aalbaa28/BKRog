import streamlit as st
import pandas as pd
import json
import os
from openai import OpenAI
from transformers import pipeline

# -------- CONFIGURAR IA


# Inicializar PandasAI con el modelo


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
        # Agregar título
        st.title("Scrim Stats")
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
        'riotIdGameName': participant['riotIdGameName'],
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

# Function to generate player summary (win rate, average stats)
def get_player_summary(df):
    player_summary = []
    
    players_of_interest = ["BKR Szygenda", "BKR Rhilech", "BKR OMON", "WD BOOSHI", "BKR Doss"]
    
    for player in players_of_interest:
        player_data = df[df['riotIdGameName'] == player]
        
        if not player_data.empty:
            total_games = len(player_data)
            wins = player_data['win'].sum()
            win_rate = (wins / total_games) * 100  # Calculate win rate
            avg_kda = player_data['kda'].mean()
            avg_deaths = player_data['deaths'].mean()
            avg_gold_per_minute = player_data['goldPerMinute'].mean()
            avg_damage_per_minute = player_data['damagePerMinute'].mean()
            avg_team_damage_percentage = player_data['teamDamagePercentage'].mean()

            player_summary.append({
                'Player': player,
                'Total Games': total_games,
                'Wins': wins,
                'WinRate': win_rate,
                'Avg KDA': avg_kda,
                'Avg Deaths': avg_deaths,
                'Avg Gold per Minute': avg_gold_per_minute,
                'Avg Damage per Minute': avg_damage_per_minute,
                'Avg Team Damage %': avg_team_damage_percentage
            })
    
    return pd.DataFrame(player_summary)

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

    # Cambiar el color de la web según la selección
    # Cambiar el color de la web según la selección
    # Cambiar el color de la web según la selección
    # Cambiar el color de la web según la selección
   # Cambiar el color de la web según la selección
    if side_filter == 'blue':
        st.markdown(
            """
            <style>
            .stApp {
               
            }
            h1, h2, h3, h4, h5, h6, strong {
                color: #0288D1;  /* Texto en azul oscuro para títulos y negritas */
            }
            .stButton>button {
                background-color: #0288D1;  /* Botón azul */
                color: white;  /* Texto del botón en blanco */
            }
            /* Estilos para las tabs */
            .stTabs [role="tab"] {
                background-color: #E0F7FA;  /* Fondo de las tabs (azul claro) */
                color: #01579B;  /* Color del texto de las tabs (azul oscuro) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            .stTabs [role="tab"][aria-selected="true"] {
                background-color: #0288D1;  /* Fondo de la tab seleccionada (azul) */
                color: white;  /* Color del texto de la tab seleccionada (blanco) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            </style>
            """, unsafe_allow_html=True
        )
    elif side_filter == 'red':
        st.markdown(
            """
            <style>
            .stApp {
                
            }
            h1, h2, h3, h4, h5, h6, strong {
                color: #C62828;  /* Texto en rojo oscuro para títulos y negritas */
            }
            .stButton>button {
                background-color: #C62828;  /* Botón rojo */
                color: white;  /* Texto del botón en blanco */
            }
            /* Estilos para las tabs */
            .stTabs [role="tab"] {
                background-color: #FFEBEE;  /* Fondo de las tabs (rojo claro) */
                color: #B71C1C;  /* Color del texto de las tabs (rojo oscuro) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            .stTabs [role="tab"][aria-selected="true"] {
                background-color: #C62828;  /* Fondo de la tab seleccionada (rojo) */
                color: white;  /* Color del texto de la tab seleccionada (blanco) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            </style>
            """, unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <style>
            .stApp {
                
            }
            h1, h2, h3, h4, h5, h6, strong {
                color: black;  /* Texto en negro para títulos y negritas */
            }
            .stButton>button {
                background-color: #2c3e50;  /* Botón gris oscuro */
                color: white;  /* Texto del botón en blanco */
            }
            /* Estilos para las tabs */
            .stTabs [role="tab"] {
                background-color: #f0f0f0;  /* Fondo de las tabs (gris claro) */
                color: #2c3e50;  /* Color del texto de las tabs (gris oscuro) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            .stTabs [role="tab"][aria-selected="true"] {
                background-color: #2c3e50;  /* Fondo de la tab seleccionada (gris oscuro) */
                color: white;  /* Color del texto de la tab seleccionada (blanco) */
                font-size: 18px;  /* Tamaño de la letra más grande */
                font-family: 'Arial', sans-serif;  /* Tipografía bonita */
                font-weight: bold;  /* Texto en negrita */
                padding: 10px 20px;  /* Espaciado interno para que se vea mejor */
                border-radius: 10px;  /* Bordes redondeados */
            }
            </style>
            """, unsafe_allow_html=True
        )

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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Top", "Jgl", "Mid", "Adc", "Supp", "By Champion", "By Player", "AI Assistant"])

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
                st.write(f"Games Played: {row['side']}")
                winrate_color = "green" if row['winrate'] >= 50 else "red"
                st.markdown(f"<p style='color:{winrate_color};'><b>Winrate:</b> {row['winrate']:.2f}%</p>", unsafe_allow_html=True)

                st.write(f"Average KDA: {row['kda']:.2f}")
                st.write(f"Average Deaths: {row['deaths']:.2f}")
                st.write(f"Gold per Minute: {row['goldPerMinute']:.2f}")
                st.write(f"Damage per Minute: {row['damagePerMinute']:.2f}")
                st.write(f"Team Damage Percentage: {row['teamDamagePercentage']*100:.2f}%")
            
            st.write("---")

with tab7:  # Assuming this is the last tab. You can rename it if needed.
    st.header("By Player")
    
    # Get player summary DataFrame
    player_summary_df = get_player_summary(combined_df)

    # Loop through each player to display a more attractive summary
    for index, row in player_summary_df.iterrows():
        # Create two columns to display the player's stats and the win rate
        col1, col2 = st.columns([2, 1])  # Adjust the column widths as needed

        with col1:
            # Estilo bonito para mostrar el nombre del jugador
            st.markdown(
                f"<h3 style='color:#7f8c8d; font-weight:bold;'>{row['Player']}</h3>", 
                unsafe_allow_html=True
            )

            st.write(f"Total Games: {row['Total Games']}")
            st.write(f"Wins: {row['Wins']}")
            
            winrate_color = "green" if row['WinRate'] >= 50 else "red"
            st.markdown(f"<p style='color:{winrate_color};'><b>Winrate</b>: {row['WinRate']:.2f}%</p>", unsafe_allow_html=True)
            
            st.write(f"Avg KDA: {row['Avg KDA']:.2f}")
            st.write(f"Avg Deaths: {row['Avg Deaths']:.2f}")
            st.write(f"Avg Gold per Minute: {row['Avg Gold per Minute']:.2f}")
            st.write(f"Avg Damage per Minute: {row['Avg Damage per Minute']:.2f}")
            st.write(f"Avg Team Damage %: {row['Avg Team Damage %']*100:.2f}%")

        with col2:
            # You can add a small image, chart or any additional info for each player
            winrate_color = "green" if row['WinRate'] >= 50 else "red"
            st.markdown(f"<span style='color:{winrate_color}; font-size: 25px;'>🔼</span>", unsafe_allow_html=True)  # You can use a simple icon or image
    
    # Show the player summary table below the details
    st.subheader("Player Summary Table")
    st.dataframe(player_summary_df.style.format({
        'WinRate': "{:.2f}%", 
        'Avg KDA': "{:.2f}", 
        'Avg Deaths': "{:.2f}", 
        'Avg Gold per Minute': "{:.2f}",
        'Avg Damage per Minute': "{:.2f}",
        'Avg Team Damage %': "{:.2f}%"
    }))

        # Function to calculate daily winrate (percentage of wins per day)
    def calculate_daily_winrate(df):
        # Ensure 'Date' is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Group by date and calculate winrate (percentage of wins per day)
        daily_winrate = df.groupby(df['Date'].dt.date)['win'].agg(
            win_rate='mean'  # Calculate the mean winrate for each day
        ).reset_index()

        # Convert win_rate to percentage
        daily_winrate['win_rate'] = daily_winrate['win_rate'] * 100
        
        return daily_winrate

    # Assume `combined_df` is the dataframe with the match data
    # Calculate the daily winrate
    daily_winrate_df = calculate_daily_winrate(combined_df)

    # Now you can plot the winrate as a bar chart or line chart
    st.subheader("Daily Winrate Comparison")

    # Plotting a bar chart
    #st.bar_chart(daily_winrate_df.set_index('Date')['win_rate'])

    # Optionally, you can use a line chart instead of a bar chart if you prefer:
    st.line_chart(daily_winrate_df.set_index('Date')['win_rate'])

    # Save the filtered results as a CSV
    if st.button('Save CSV'):
        csv_file = 'filtered_matchups.csv'
        combined_df.to_csv(csv_file, index=False)
        st.success(f"CSV saved as {csv_file}")

with tab8:
    st.header("🤖 AI Assistant - Ask about Scrim Stats")
    # Cargar un modelo de Hugging Face
    # Convertir el DataFrame a texto para usarlo como contexto
    data_summary = combined_df.to_string()

    # Cargar un modelo de Hugging Face (usamos GPT-2 como ejemplo)
    generator = pipeline("text-generation", model="gpt2")

    # Crear un campo de entrada para que el usuario haga preguntas
    user_input = st.chat_input("Ask me anything about scrim data...")

    # Mostrar el historial del chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Procesar la pregunta y mostrar la respuesta
    if user_input:
        # Mostrar la pregunta del usuario
        with st.chat_message("user"):
            st.markdown(user_input)

        # Agregar la pregunta al historial del chat
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Generar la respuesta usando el modelo de Hugging Face
        try:
            prompt = f"You are a helpful assistant that answers questions about scrim data. Here is the data:\n{data_summary}\n\nQuestion: {user_input}\nAnswer:"
            response = generator(
                prompt,
                max_length=200,  # Aumenta la longitud máxima
                max_new_tokens=50,  # Limita la longitud de la respuesta
                num_return_sequences=1
            )
            assistant_response = response[0]["generated_text"].split("Answer:")[-1].strip()
        except Exception as e:
            assistant_response = "Sorry, I'm currently unable to process your request. Please try again later."
            st.error(f"Error: {e}")

        # Mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            st.markdown(assistant_response)

        # Agregar la respuesta al historial del chat
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

