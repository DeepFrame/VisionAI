import os
import sys

# This script performs a lightweight, non-intrusive healthcheck.
# It verifies the application is ready to run by checking for the presence of
# key files and directories, without connecting to the database or network.

def run_healthcheck():
    """
    Checks for the existence of required files and directories.
    Returns 0 for success, 1 for failure.
    """
    try:
        # Check for the main application entry point.
        if not os.path.exists("../main.py"):
            print("Healthcheck failed: main.py not found.")
            return 1
            
        # Check for the logs directory. This also verifies the host-mounted volume.
        if not os.path.isdir("../logs"):
            print("Healthcheck failed: logs directory not found.")
            return 1
            
        # All checks passed.
        print("Healthcheck passed: Application is ready.")
        return 0
        
    except Exception as e:
        print(f"Healthcheck failed due to an unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_healthcheck()
    sys.exit(exit_code)
