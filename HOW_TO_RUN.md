# How to Run the Chess Game

This guide provides detailed instructions for running the chess game both locally and as a web demo.

## Table of Contents
- [Running Locally](#running-locally)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Game](#running-the-game)
- [Web Demo](#web-demo)
  - [Play Online](#play-online)
  - [Running the Web Demo Locally](#running-the-web-demo-locally)
- [Troubleshooting](#troubleshooting)
  - [Dependency Issues](#dependency-issues)
  - [Platform-Specific Issues](#platform-specific-issues)
  - [Web Demo Issues](#web-demo-issues)

## Running Locally

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

### Installation

1. Clone or download the repository:
   ```
   git clone https://github.com/isaackorda/CompSciNEA_IK.git
   cd CompSciNEA_IK
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   **Note for non-macOS users**: The requirements.txt file includes macOS-specific TensorFlow packages. If you're on Windows or Linux, you can install the standard TensorFlow package instead:
   ```
   pip install pygame>=2.6.1 board>=1.0 chess>=1.10.0 tqdm>=4.66.6 numpy>=1.26.1 tensorflow==2.12.0 keras==2.12.0
   ```

### Running the Game

To run the game locally:

```
python main.py
```

This will launch the chess game with a graphical user interface. You'll see a main menu with options to start the game or quit.

**Controls:**
- Click on a piece to select it
- Click on a highlighted square to move the piece
- Press 'T' to change the theme
- Press 'R' to reset the game

## Web Demo

### Play Online

You can play the chess game directly in your web browser without installing anything:

[Click here to play the chess game in your browser](https://isaackorda.github.io/CompSciNEA_IK/)

The web demo works best in modern browsers like Chrome, Firefox, Edge, or Safari.

### Running the Web Demo Locally

If you want to build and run the web demo locally:

1. Install the required dependencies and Pygbag:
   ```
   pip install -r requirements.txt
   pip install pygbag
   ```

2. Run the build script:
   ```
   python build_web_demo.py
   ```

   This will:
   - Build the web demo using Pygbag
   - Start a local web server on port 8000
   - Open your default browser to play the game

3. Additional options:
   ```
   python build_web_demo.py --help
   ```

   - `--port PORT`: Specify a different port (default: 8000)
   - `--no-browser`: Don't open the browser automatically
   - `--build-only`: Only build the web demo, don't run it

4. To stop the web server, press Ctrl+C in the terminal where you ran the build script.

## Troubleshooting

### Dependency Issues

**Missing dependencies:**
```
ModuleNotFoundError: No module named 'xyz'
```

Solution: Make sure you've installed all required dependencies:
```
pip install -r requirements.txt
```

**TensorFlow issues:**
- If you're on Windows or Linux and see errors related to tensorflow-metal or tensorflow-macos, install the standard TensorFlow package instead:
  ```
  pip install tensorflow==2.12.0
  ```

**Pygame installation issues:**
- On some Linux distributions, you may need to install additional system packages:
  ```
  sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev
  ```

### Platform-Specific Issues

**macOS:**
- If you get a security warning when running the game, go to System Preferences > Security & Privacy and click "Open Anyway"
- For Apple Silicon Macs, make sure you're using Python for Apple Silicon

**Windows:**
- If you get "DLL load failed" errors with Pygame, try reinstalling Pygame:
  ```
  pip uninstall pygame
  pip install pygame --pre
  ```

**Linux:**
- If the game window doesn't appear or crashes, check that you have the required X11 libraries:
  ```
  sudo apt-get install libx11-dev
  ```

### Web Demo Issues

**Pygbag installation fails:**
- Make sure you have a compatible Python version (3.8 or higher)
- Try installing with pip directly:
  ```
  python -m pip install pygbag
  ```

**Web demo doesn't load:**
- Check that you're using a modern browser (Chrome, Firefox, Edge, or Safari)
- Try clearing your browser cache
- Check the browser console for any JavaScript errors

**Web demo is slow:**
- The web demo converts Python to WebAssembly, which may run slower than the native version
- Try closing other browser tabs and applications to free up resources

If you encounter any other issues, please report them on the [GitHub repository](https://github.com/isaackorda/CompSciNEA_IK/issues).