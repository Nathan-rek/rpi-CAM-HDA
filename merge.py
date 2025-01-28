import os
import subprocess

# Nom du fichier final
output_filename = "merged_output.mp4"

# Trouver tous les fichiers commençant par "output_" et ayant l'extension ".mp4"
video_files = sorted([f for f in os.listdir('.') if f.startswith('output_') and f.endswith('.mp4')])

if not video_files:
    print("Aucun fichier vidéo 'output_' trouvé.")
else:
    # Nom du fichier temporaire pour la liste de lecture
    playlist_filename = "file_list.txt"

    # Créer la liste de lecture pour ffmpeg
    with open(playlist_filename, 'w') as playlist_file:
        for video in video_files:
            playlist_file.write(f"file '{video}'\n")

    # Commande ffmpeg pour concaténer les vidéos
    ffmpeg_command = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', playlist_filename, '-c', 'copy', output_filename
    ]

    try:
        # Exécuter la commande ffmpeg
        subprocess.run(ffmpeg_command, check=True)
        print(f"Vidéo combinée créée avec succès : {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de ffmpeg : {e}")
    finally:
        # Nettoyer le fichier temporaire
        if os.path.exists(playlist_filename):
            os.remove(playlist_filename)
