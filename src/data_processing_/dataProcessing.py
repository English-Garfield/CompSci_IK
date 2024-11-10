"""
processing the Data pre NN
"""
import os
import time

from chess import pgn
from tqdm import tqdm

file_path = '../../assets/ChessData'
Large_file_path = '../assets/ChessData/LargeData'
i = 1


def load_pgn_files(file_path):
    games = []
    with open(file_path, 'r') as pgn_file:
        while True:
            game = pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games


def storingData(data):
    file = open("ProcessedChessData.txt", "a")
    file.write(data)
    file.close()

    print("Data written to file")


print("Start of file parsing")
Program_start_time = time.time()

files = [file for file in os.listdir(file_path) if file.endswith('.pgn')]
games = []

for file in tqdm(files, colour='green'):
    games.extend(load_pgn_files(f"{file_path}/{file}"))
    i += 1

storingData(str(games))  # storing the data in a text file

for file in tqdm(files, colour='blue'):
    games.extend(load_pgn_files(f"{Large_file_path}/{file}"))
    i += 1

storingData(str(games))  # storing the data in a text file

end_time = time.time()
print("\n")
print(f"End of file parsing. Time taken: {end_time - Program_start_time} seconds")


