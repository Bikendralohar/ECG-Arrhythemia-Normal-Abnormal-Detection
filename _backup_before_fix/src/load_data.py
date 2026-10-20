"""
Part 1: Download MIT-BIH Arrhythmia Database records.

Run this once to download all records you need locally.
"""

import wfdb
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "mitbih_data")
os.makedirs(DATA_DIR, exist_ok=True)

# Full list of standard MIT-BIH record names (patient IDs).
# You can start with a small subset for testing, then use the full list.
ALL_RECORDS = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]

# For quick testing, start with a small subset. Switch to ALL_RECORDS
# once you're ready to build the full training dataset.
TEST_RECORDS = ['100', '101', '102']


def available_records(data_dir=DATA_DIR):
    """Return complete MIT-BIH records in numeric order."""
    record_names = []
    for filename in os.listdir(data_dir):
        if not filename.endswith('.hea'):
            continue

        record_name = filename[:-4]
        required_files = (
            f'{record_name}.hea',
            f'{record_name}.dat',
            f'{record_name}.atr',
        )
        if all(os.path.isfile(os.path.join(data_dir, required))
               for required in required_files):
            record_names.append(record_name)

    return sorted(record_names, key=lambda name: int(name))


def download_records(record_list, dl_dir=DATA_DIR):
    """Download a list of MIT-BIH records from PhysioNet."""
    print(f"Downloading {len(record_list)} record(s) to {dl_dir} ...")
    wfdb.dl_database('mitdb', dl_dir=dl_dir, records=record_list)
    print("Download complete.")


if __name__ == "__main__":
    download_records(TEST_RECORDS)
