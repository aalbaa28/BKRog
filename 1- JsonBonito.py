import json

# Función para formatear un JSON bonito
def formatear_json(json_input):
    # Si es una ruta de archivo, lo cargamos
    if isinstance(json_input, str):
        with open(json_input, 'r') as file:
            data = json.load(file)
    else:
        # Si ya es un objeto JSON cargado (diccionario/lista), lo usamos directamente
        data = json_input

    # Convertir el JSON a un formato bonito (con indentación)
    return json.dumps(data, indent=4, separators=(',', ': '))

# Ejemplo de uso con un archivo JSON
archivo_entrada = 'd:\\Scrims\\March 18\\G4.json'  # Ruta del archivo JSON de entrada
archivo_salida = 'd:\\Scrims\\March 18\\G4f.json'  # Ruta para guardar el archivo bonito

# Formatear el JSON y mostrarlo en consola
print(formatear_json(archivo_entrada))

# También puedes guardar el resultado en un archivo
with open(archivo_salida, 'w') as file:
    file.write(formatear_json(archivo_entrada))
