from pathlib import Path

SAMPLING_RATE = 1000  # Hz

NUM_CHANNELS = 4
CHUNK_SIZE = 10

VCC = 3.3
SAMPLING_RESOLUTION = 2 ** 10
SENSOR_GAIN = 1009


def project_root() -> Path:
    Path(__file__).parent.mkdir(exist_ok=True, parents=True)
    return Path(__file__).parent


def settings_file() -> Path:
    p = project_root()
    if "Temp" in p.parts:
        f = (
                p.parents[len(p.parts) - 2 - p.parts.index("Temp")]
                / "emg_interface"
                / "emg_interface.ini"
        )
    else:
        f = project_root() / "emg_interface.ini"

    f.parent.mkdir(exist_ok=True, parents=True)
    return f
