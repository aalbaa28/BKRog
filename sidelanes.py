import os
import json
from datetime import datetime, timedelta

def calculate_szygenda_winrate_last_14_days(folder_path):
    # Contadores para las estadísticas
    blue_games = 0
    blue_wins = 0
    red_games = 0
    red_wins = 0
    
    # Calcular la fecha límite (hoy - 14 días)
    cutoff_date = datetime.now() - timedelta(days=14)
    
    # Iterar sobre todos los archivos en la carpeta
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Verificar si el juego es de los últimos 14 días
                game_date = datetime.fromtimestamp(json_data['gameCreation'] / 1000)
                if game_date < cutoff_date:
                    continue  # Saltar juegos anteriores al período
                
                # Buscar a Szygenda en los participantes
                for participant in json_data['participants']:
                    if participant['riotIdGameName'] == "BKR Szygenda":
                        side = 'blue' if participant['teamId'] == 100 else 'red'
                        win = participant['win']
                        
                        if side == 'blue':
                            blue_games += 1
                            if win:
                                blue_wins += 1
                        else:
                            red_games += 1
                            if win:
                                red_wins += 1
                        break  # Salir del bucle una vez encontrado
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
    
    # Calcular winrates
    blue_winrate = (blue_wins / blue_games * 100) if blue_games > 0 else 0
    red_winrate = (red_wins / red_games * 100) if red_games > 0 else 0
    
    # Imprimir resultados
    print(f"Stats for BKR (last 14 days):")
    print(f"Blue Side: {blue_wins} wins in {blue_games} games ({blue_winrate:.2f}%)")
    print(f"Red Side: {red_wins} wins in {red_games} games ({red_winrate:.2f}%)")
    print(f"Total games: {blue_games + red_games}")
    
    return {
        'blue_side': {'games': blue_games, 'wins': blue_wins, 'winrate': blue_winrate},
        'red_side': {'games': red_games, 'wins': red_wins, 'winrate': red_winrate},
        'total_games': blue_games + red_games,
        'period': 'last_14_days'
    }

# Uso de la función
folder_path = "d:/Scrims/March 18"  # Reemplaza con tu ruta real
stats = calculate_szygenda_winrate_last_14_days(folder_path)