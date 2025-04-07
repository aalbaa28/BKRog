import json
import os
import re  # <-- Añade esto junto con tus otros imports
import google.generativeai as genai
import numpy as np
import pandas as pd
import streamlit as st
import requests
from bs4 import BeautifulSoup

# -------- CONFIGURAR IA


if st.button("Recargar datos JSON"):
    st.cache_data.clear()


# Inicializar PandasAI con el modelo

@st.cache_data  # Caching the data loading to avoid repeated processing
def load_json_data(folder):
    combined_data = []
    try:
        json_files = [file for file in os.listdir(folder) if file.endswith('.json')]
        for file in json_files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                combined_data.append(json_data)
        return combined_data  # Solo devuelve los datos, sin UI
    except Exception as e:
        return str(e) 

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
        'controlWards': challenges['controlWardsPlaced']
    }
    return data



# Function to convert match data into a DataFrame
def get_matchup_data(json_data):
    matchups = []

    # List of players we are interested in
    players_of_interest = ["BKR Szygenda", "BKR Rhilech", "BKR OMON", "WD BOOSHI", "BKR Doss", "BKR BOOSHI"]

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
            elif participant['riotIdGameName'] == "WD BOOSHI" or participant['riotIdGameName'] == "BKR BOOSHI":
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
        'win': 'sum',  # Sum of wins to calculate WR
        'controlWards': 'mean'
    }).reset_index()

    # Calculate Win Rate (WR) as (wins / total games) * 100
    avg_df['winrate'] = (avg_df['win'] / avg_df['side']) * 100

    # Sort by total games played (side count) and then by KDA for better display
    avg_df = avg_df.sort_values(by='side', ascending=False)

    return avg_df

# Function to generate player summary (win rate, average stats)
def get_player_summary(df):
    player_summary = []

    # Lista de jugadores de interés (sin incluir 'WD BOOSHI' porque se combinará con 'BKR BOOSHI')
    players_of_interest = ["BKR Szygenda", "BKR Rhilech", "BKR OMON", "BKR BOOSHI", "BKR Doss"]

    for player in players_of_interest:
        if player == "BKR BOOSHI":
            # Combina los datos de "WD BOOSHI" y "BKR BOOSHI"
            player_data = df[df['riotIdGameName'].isin(["WD BOOSHI", "BKR BOOSHI"])]
        else:
            # Obtiene los datos para los demás jugadores
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
            avg_control_wards = player_data['controlWards'].mean()

            # Si es BKR BOOSHI, aseguramos que el nombre mostrado sea "BKR BOOSHI"
            player_name = "BKR BOOSHI" if player == "BKR BOOSHI" else player

            player_summary.append({
                'Player': player_name,  # Mostramos "BKR BOOSHI" cuando sea el caso
                'Total Games': total_games,
                'Wins': wins,
                'WinRate': win_rate,
                'Avg KDA': avg_kda,
                'Avg Deaths': avg_deaths,
                'Avg Gold per Minute': avg_gold_per_minute,
                'Avg Damage per Minute': avg_damage_per_minute,
                'Avg Team Damage %': avg_team_damage_percentage,
                'Avg Control Wards': avg_control_wards
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

        # Inicializar session_state si no existe
        if "filters_applied" not in st.session_state:
            st.session_state.filters_applied = False
            st.session_state.filtered_df = combined_df  # Para evitar cálculos innecesarios

        # Filter by side (blue or red)

        
    # Función para calcular winrate por side (debe ir antes de su uso)
    def calculate_winrate(players_data):
        """
        Calcula el winrate global por lado (blue/red) para todos los jugadores.
        Args:
            players_data: Lista de diccionarios con datos de partidas (debe contener 'side' y 'win')
        Returns:
            blue_winrate: Winrate en % para blue side
            red_winrate: Winrate en % para red side
            (También imprime los resultados)
        """
        blue_wins = 0
        red_wins = 0
        blue_games = 0
        red_games = 0

        for player in players_data:
            if player['side'] == 'blue':
                blue_games += 1
                if player['win']:
                    blue_wins += 1
            else:  # 'red' side
                red_games += 1
                if player['win']:
                    red_wins += 1

        blue_winrate = round((blue_wins / blue_games) * 100, 2) if blue_games > 0 else 0
        red_winrate = round((red_wins / red_games) * 100, 2) if red_games > 0 else 0

        st.write(f"📊 Winrate Global por Side:")
        st.write(f"🔵 Blue Side: {blue_winrate}% ({blue_wins}/{blue_games} partidas ganadas)")
        st.write(f"🔴 Red Side: {red_winrate}% ({red_wins}/{red_games} partidas ganadas)")
        
        return blue_winrate, red_winrate

    # Botón para calcular winrate por side
    if st.button("📊 Calcular Winrate por Side"):
        if 'combined_df' in globals():
            players_data = combined_df.to_dict('records')
            calculate_winrate(players_data)
        else:
            st.warning("Primero carga los datos JSON")

    # Inicializar session_state si no existe
    if "filters_applied" not in st.session_state:
        st.session_state.filters_applied = False

    # Sidebar filters sin recargar la página
    with st.sidebar:
        side_filter = st.selectbox("Filter by side", ['All', 'blue', 'red'], key="side_filter")
        champion_list = sorted(combined_df['championName'].unique().tolist())
        champion_list.insert(0, "All")
        champion_filter = st.selectbox("Filter by champion", champion_list, key="champion_filter")
        start_date = st.date_input("Start Date", pd.to_datetime('2025-03-01'), key="start_date")
        end_date = st.date_input("End Date", pd.to_datetime('2025-04-30'), key="end_date")

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

            if st.button("Apply Filters"):
                st.session_state.filters_applied = True  # Marcar que los filtros deben aplicarse

    # Solo aplicar filtros si el botón ha sido presionado
    if st.session_state.filters_applied:
        start_date = pd.to_datetime(st.session_state.start_date)
        end_date = pd.to_datetime(st.session_state.end_date).replace(hour=23, minute=59, second=59)

        if st.session_state.side_filter != 'All':
            combined_df = combined_df[combined_df['side'] == st.session_state.side_filter]

        if st.session_state.champion_filter != "All":
            combined_df = combined_df[combined_df['championName'] == st.session_state.champion_filter]

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
                    st.write(f"Control Wards placed: {row['controlWards']}")
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
                st.write(f"Control Wards Placed: {row['controlWards']:.2f}")

            st.write("---")

with tab7:  # Assuming this is the last tab. You can rename it if needed.
    st.header("By Player")

    # Get player summary DataFrame
    player_summary_df = get_player_summary(combined_df)

        # Show the player summary table below the details
    st.subheader("Player Summary Table")

    all_columns = player_summary_df.columns.tolist()

    # Selector para elegir qué columnas mostrar
    selected_columns = st.multiselect("Select columns to display:", all_columns, default=all_columns)

    # Botón para actualizar la tabla
    if st.button("Show Data"):
        st.data_editor(player_summary_df[selected_columns])

    


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



api_key = st.secrets["api_key"]
# Configuration
genai.configure(api_key=st.secrets["api_key"])
model = genai.GenerativeModel('gemini-1.5-pro')
        

# --------------------------------------
@st.cache_data(ttl=3600)  # Cache por 1 hora
def get_opgg_grandmaster_stats(champion, position):
    """
    Obtiene stats de Grandmaster+ desde OP.GG
    champion: nombre en inglés (ashe, zed, ksante)
    position: top, jungle, mid, adc, support
    """
    url = f"https://op.gg/api/v1.0/internal/bypass/champions/{champion}/statistics?position={position.lower()}&tier=grandmaster&region=world"
    
    try:
        response = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "en-US,en;q=0.9"
            },
            timeout=10
        )
        data = response.json()
        
        return {
            'kda': [
                data['stats']['general']['kill'],
                data['stats']['general']['death'],
                data['stats']['general']['assist']
            ],
            'gpm': data['stats']['general']['gold_per_min'],
            'dpm': data['stats']['general']['damage_per_min'],
            'win_rate': data['stats']['general']['win_rate'],
            'pick_rate': data['stats']['general']['pick_rate'],
            'games_analyzed': data['stats']['general']['total'],
            'source': 'OP.GG (Grandmaster+)'
        }
    except Exception as e:
        return {"error": f"Error API OP.GG: {str(e)} | URL: {url}"}

# 2. Tu función para calcular promedios
def calculate_player_stats(df, champion, position):
    """Usa TU función para calcular stats del jugador"""
    avg_df = calculate_average_by_champion(
        df[(df['championName'] == champion) & (df['Position'] == position)]
    )
    
    if avg_df.empty:
        return None
    
    return avg_df.iloc[0].to_dict()

# 3. Interfaz Streamlit
st.title("🏆 Mi Rendimiento vs Grandmaster+ (OP.GG)")

df = combined_df
st.session_state.df = df

# Filtros
if 'df' in st.session_state:
    df = st.session_state.df
    
    col1, col2 = st.columns(2)
    with col1:
        position = st.selectbox(
            "Posición",
            options=sorted(df['Position'].unique()),
            format_func=lambda x: x.capitalize()
        )
    with col2:
        champion = st.selectbox(
            "Campeón",
            options=sorted(df[df['Position'] == position]['championName'].unique())
        )

    # Botón de análisis
    if st.button("🔍 Comparar con Grandmaster+"):
        with st.spinner("Analizando..."):
            try:
                # Obtener stats del jugador
                player_stats = calculate_player_stats(df, champion, position)
                if not player_stats:
                    st.error("No tienes datos con esta combinación")
                    st.stop()
                
                # Obtener stats GM+
                gm_stats = get_opgg_grandmaster_stats(champion, position)
                if "error" in gm_stats:
                    st.error(gm_stats["error"])
                    st.stop()
                
                # Tabla comparativa
                st.subheader(f"📊 {champion} - {position}")
                comparison = pd.DataFrame({
                    'Métrica': ['KDA', 'GPM', 'DPM', 'Win Rate', 'Pick Rate', 'Partidas'],
                    'Tú': [
                        f"{player_stats['kda']:.2f}",
                        f"{player_stats['goldPerMinute']:.1f}",
                        f"{player_stats['damagePerMinute']:.1f}",
                        f"{player_stats['winrate']:.1f}%",
                        "N/A",
                        int(player_stats['side'])
                    ],
                    'Grandmaster+': [
                        f"{gm_stats['kda'][0]:.1f}/{gm_stats['kda'][1]:.1f}/{gm_stats['kda'][2]:.1f}",
                        f"{gm_stats['gpm']:.1f}",
                        f"{gm_stats['dpm']:.1f}",
                        f"{gm_stats['win_rate']:.1f}%",
                        f"{gm_stats['pick_rate']:.1f}%",
                        gm_stats['games_analyzed']
                    ]
                })
                
                st.dataframe(
                    comparison.style.highlight_max(axis=1, color='#90EE90'),
                    hide_index=True,
                    width=800
                )
                
                # Análisis de IA
                st.subheader("🧠 Análisis Comparativo")
                prompt = f"""Compara estos stats de {champion} en {position}:
                - KDA: {player_stats['kda']:.2f} (Tú) vs {gm_stats['kda'][0]:.1f}/{gm_stats['kda'][1]:.1f}/{gm_stats['kda'][2]:.1f} (GM+)
                - GPM: {player_stats['goldPerMinute']:.1f} vs {gm_stats['gpm']:.1f}
                - DPM: {player_stats['damagePerMinute']:.1f} vs {gm_stats['dpm']:.1f}
                - Win Rate: {player_stats['winrate']:.1f}% vs {gm_stats['win_rate']:.1f}%
                - Partidas analizadas: {player_stats['side']} (Tú) vs {gm_stats['games_analyzed']} (GM+)
                
                Genera un análisis en español con:
                1. Diagnóstico general (1 oración)
                2. 2 aspectos destacados positivos
                3. 3 áreas clave de mejora
                4. 1 consejo estratégico específico para {position}
                """
                response = model.generate_content(prompt)
                st.write(response.text)
                
                # Mostrar imagen del campeón si existe
                if 'championImage' in player_stats and pd.notna(player_stats['championImage']):
                    st.image(player_stats['championImage'], width=150)
                
            except Exception as e:
                st.error(f"Error inesperado: {str(e)}")