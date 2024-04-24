def translater():
    fenArray = []
    fenString = input("What is the FEN code: ")

    for char in fenString:
        fenArray.append(char)

    for i in (len(fenArray) + 1):
        piece = fenArray[0]

        if piece.isupper():
            # piece is white
            print("piece is white")

        elif piece.islower():
            # piece is black
            print("piece is black")

        else:
            # number
            print("piece is black")

        del (fenArray[0])