#!/usr/bin/env python3
"""
Build script for the web demo.
This script uses Pygbag to build and run the web demo locally.
"""

import os
import sys
import subprocess
import webbrowser
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description='Build and run the web demo')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the web server on')
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
    parser.add_argument('--build-only', action='store_true', help='Only build the web demo, do not run it')
    args = parser.parse_args()

    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the script directory
    os.chdir(script_dir)
    
    print("Building web demo...")
    
    # Build the web demo using Pygbag
    build_cmd = [
        sys.executable, "-m", "pygbag", 
        "--ume_block", "0",  # Don't block the UI while loading
        "--app_name", "Chess with Deep Learning Neural Network",
        "--title", "Chess with Deep Learning Neural Network - Web Demo",
        "--icon", "assets/images/black-knight.png",  # Assuming this path exists
        "--html", "index.html",  # Use our custom HTML template
        "--package", "web_build",  # Output directory
        "--build",  # Build only, don't serve
        "."  # Current directory (root of the project)
    ]
    
    try:
        subprocess.run(build_cmd, check=True)
        print("Web demo built successfully!")
        
        if not args.build_only:
            # Serve the web demo
            port = args.port
            print(f"Starting web server on port {port}...")
            
            # Run the web server in a separate process
            server_cmd = [
                sys.executable, "-m", "http.server", 
                "--directory", "web_build", 
                str(port)
            ]
            
            server_process = subprocess.Popen(server_cmd)
            
            # Open the browser
            if not args.no_browser:
                url = f"http://localhost:{port}"
                print(f"Opening {url} in the default browser...")
                time.sleep(1)  # Give the server a moment to start
                webbrowser.open(url)
            
            print("Press Ctrl+C to stop the server...")
            try:
                # Keep the server running until the user presses Ctrl+C
                server_process.wait()
            except KeyboardInterrupt:
                print("Stopping server...")
            finally:
                server_process.terminate()
                server_process.wait()
                
        print("Done!")
        
    except subprocess.CalledProcessError as e:
        print(f"Error building web demo: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())