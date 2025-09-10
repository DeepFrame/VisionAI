import os
import sys

def run_healthcheck():
    """
    Checks for the existence of required files and directories.
    Returns 0 for success, 1 for failure.
    """
    try:
        if not os.path.exists("../main.py"):
            print("Healthcheck failed: main.py not found.")
            return 1
            
        if not os.path.isdir("../logs"):
            print("Healthcheck failed: logs directory not found.")
            return 1
            
        print("Healthcheck passed: Application is ready.")
        return 0
        
    except Exception as e:
        print(f"Healthcheck failed due to an unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = run_healthcheck()
    sys.exit(exit_code)
