import time
import os
import sys
from datetime import datetime
import subprocess

# Ensure Python path includes the root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def wait_until_next_15m_candle():
    """
    Calculates the seconds remaining until the next 15-minute mark
    (e.g., :00, :15, :30, :45) and sleeps until that time.
    """
    now = datetime.now()

    # Calculate how many minutes past the hour we are
    minute = now.minute

    # Find the next 15-minute interval
    if minute < 15:
        next_minute = 15
    elif minute < 30:
        next_minute = 30
    elif minute < 45:
        next_minute = 45
    else:
        # Next interval is the start of the next hour
        next_minute = 60

    # Calculate seconds until the target time
    seconds_to_wait = (next_minute - minute) * 60 - now.second

    # Add a small buffer (e.g., 5 seconds) to ensure the candle is fully closed
    # on the server side and new data is available to fetch.
    seconds_to_wait += 5

    print(f"[{now.strftime('%H:%M:%S')}] Waiting {seconds_to_wait} seconds until the next 15m candle close...")
    time.sleep(seconds_to_wait)

def start_automated_loop():
    print("==================================================")
    print("      STARTING FULLY AUTOMATED TRADING BOT        ")
    print("==================================================")
    print("Press Ctrl+C to stop the bot at any time.\n")

    # Get the absolute path to the main execution script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_script_path = os.path.join(script_dir, "src", "main.py")

    while True:
        try:
            wait_until_next_15m_candle()

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Triggering Trading Cycle...")

            # Execute the main.py script in a subprocess
            # We set the PYTHONPATH to ensure imports work correctly
            env = os.environ.copy()
            env["PYTHONPATH"] = script_dir

            result = subprocess.run(
                [sys.executable, main_script_path],
                env=env,
                capture_output=True,
                text=True
            )

            # Print the output from the bot cycle
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(f"ERRORS:\n{result.stderr}")

        except KeyboardInterrupt:
            print("\nAutomated bot stopped by user. Exiting gracefully.")
            break
        except Exception as e:
            print(f"\nAn error occurred in the automation loop: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)

if __name__ == "__main__":
    start_automated_loop()
