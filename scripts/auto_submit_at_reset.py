#!/usr/bin/env python3
"""
Automated Submission Watchdog for Virtual Cell Challenge (VCC 2026)
Monitors time until 00:00 UTC daily allowance reset and submits immediately upon reset.
"""

import time
import datetime
import subprocess
import os
import sys

def run_watchdog():
    vcc_bin = os.path.abspath("./vcc_env/bin/vcc")
    vcc_file = os.path.abspath("submissions/state_of_the_art_v5.vcc")
    log_file = os.path.abspath("submissions/v5_submission_result.log")

    if not os.path.exists(vcc_file):
        print(f"Error: Submission file {vcc_file} not found!", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("VCC 2026 Automated Submission Watchdog Activated")
    print(f"Target Submission: {vcc_file}")
    print(f"Log Output: {log_file}")
    print("=" * 60)

    while True:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        midnight_utc = (now_utc + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        remaining = (midnight_utc - now_utc).total_seconds()

        if remaining <= 5: # Within 5 seconds of 00:00 UTC
            print(f"[{datetime.datetime.now()}] 00:00 UTC reached! Triggering submission...")
            time.sleep(max(1.0, remaining + 1.0)) # Wait until 00:00:01 UTC

            cmd = [
                vcc_bin, "submit", vcc_file,
                "-m", "SOTA Network Regulon Model v5",
                "-d", "100% STRING physical & functional interactors + CollecTRI signed TF regulons + context basal single-cell preservation",
                "--wait"
            ]

            print(f"Executing: {' '.join(cmd)}")
            with open(log_file, "w") as log_f:
                proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)

            print(f"Submission process finished with exit code {proc.returncode}")
            with open(log_file, "r") as log_f:
                print(log_f.read())
            break
        else:
            hours, rem = divmod(int(remaining), 3600)
            minutes, secs = divmod(rem, 60)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Time to 00:00 UTC reset: {hours:02d}h {minutes:02d}m {secs:02d}s. Sleeping...")
            # Sleep in intervals of up to 300 seconds
            sleep_time = min(300.0, max(10.0, remaining - 10.0))
            time.sleep(sleep_time)

if __name__ == "__main__":
    run_watchdog()
