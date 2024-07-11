import os

def generate_file_list(directory, output_file):
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.jpeg'):
                file.write(f"{directory}/{filename}\n")

# Define los directorios y los archivos de salida
train_directory = 'data_yolo/train'
validation_directory = 'data_yolo/validation'
train_output = 'data_yolo/train.txt'
validation_output = 'data_yolo/validation.txt'

# Genera los archivos de lista
generate_file_list(train_directory, train_output)
generate_file_list(validation_directory, validation_output)

print(f"Archivos {train_output} y {validation_output} generados exitosamente.")
