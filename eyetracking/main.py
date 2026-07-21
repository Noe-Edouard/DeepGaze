# Copyright (c) 1996-2024, SR Research Ltd., All Rights Reserved
#
# For use by SR Research licencees only. Redistribution and use in source
# and binary forms, with or without modification, are NOT permitted.
#
# DESCRIPTION:
# Refactored PsychoPy + EyeLink script for free-viewing picture presentation.
#
# This version keeps the existing EyeLink logic but also makes the output
# easier to parse with the current processing pipeline:
# - fixed image presentation duration
# - random central fixation duration between 800 and 1200 ms
# - subset selection based on participant ID
# - no skip with spacebar
# - aspect-ratio-preserving image display inside a configurable maximum area
# - tracker calibration and gaze data remain in full-screen coordinates
# - periodic drift check every 25 images
# - periodic full recalibration every 50 images
# - participant-specific deterministic shuffle within each subset
# - parser-friendly EyeLink messages: NUM_IMAGE, Affichage_Image, raw image name
# - local EDF file name compatible with Extractor ID-(\d+) convention
# - modality-aware results folders (results/visible or results/infrared)
# - optional TEST_MODE using test/tables, test/images and results/test
# - screen size detected automatically at runtime and stored in EDF trial vars
#


from __future__ import division, print_function

import os
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters, digits

import pandas as pd
import pylink
from PIL import Image
from src.EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
from psychopy import core, event, gui, logging, monitors, visual
from psychopy.hardware import keyboard


# -----------------------------------------------------------------------------
# Global settings
# -----------------------------------------------------------------------------

logging.console.setLevel(logging.CRITICAL)

DUMMY_MODE = True
FULL_SCREEN = True
TEST_MODE = False

# Maximum drawing area for the stimulus, in pixels.
# The real screen size is detected automatically at runtime.
STIMULUS_MAX_SIZE = (1200, 800)

IMG_TIME = 3.0
FIXATION_MIN_TIME = 0.8
FIXATION_MAX_TIME = 1.2
CENTRAL_RECALIBRATION_EVERY = 25
FULL_RECALIBRATION_EVERY = 50

TABLES_DIR = Path("test/tables") if TEST_MODE else Path("tables")
STIMULI_ROOT_DIR = Path("test/stimuli") if TEST_MODE else Path("stimuli")
RESULTS_BASE_DIR = Path("results/test") if TEST_MODE else Path("results")

PARTICIPANTS_TABLE = TABLES_DIR / "participants.csv"
STIMULI_TABLE = TABLES_DIR / "metadata.csv"
CALIBRATION_TARGET = Path("src/fixTarget.bmp")
HOST_IP = '100.1.1.1'
MONITOR_NAME = 'myMonitor'
MONITOR_WIDTH_CM = 53.5
MONITOR_DISTANCE_CM = 70.0

MODALITY_TO_IMAGE_FOLDER = {
    "RGB": "visible",
    "IR": "infrared",
    "VISIBLE": "visible",
    "infrared": "infrared",
}

# Runtime globals used by helper functions.
use_retina = False
win = None
genv = None
session_folder = None
session_identifier = None
edf_file = None
screen_width = None
screen_height = None
stimulus_max_width = None
stimulus_max_height = None
participant_id_global = None
current_subset_global = None
current_modality_global = None
current_modality_folder_global = None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class Geometry:
    """Displayed stimulus geometry in full-screen pixel coordinates."""
    img_width: int
    img_height: int
    center_x: int
    center_y: int
    left: int
    top: int
    right: int
    bottom: int


@dataclass
class Trial:
    """Container for one trial."""
    condition: str
    display_image: str
    parser_image_name: str
    num_image: int
    display_image_path: str
    display_image_relpath: str


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------

def switch_to_script_folder():
    """Run the script from its own folder so relative paths stay stable."""
    script_path = os.path.dirname(sys.argv[0])
    if script_path:
        os.chdir(script_path)


def ask_retina_mode_if_needed():
    """Ask for retina mode on macOS only."""
    global use_retina

    if 'Darwin' not in platform.system():
        use_retina = False
        return

    dlg = gui.Dlg(title="Display configuration")
    dlg.addText("Screen type")
    dlg.addText("Select the monitor resolution mode used for this session.")
    dlg.addField(
        "Mode",
        choices=[
            "High Resolution (Retina, 2k, 4k, 5k)",
            "Standard Resolution (HD or lower)",
        ],
    )
    result = dlg.show()
    if not dlg.OK:
        print('user cancelled')
        core.quit()
        sys.exit()

    use_retina = result[0] == "High Resolution (Retina, 2k, 4k, 5k)"


def ask_participant_and_edf_name():
    """Get a valid participant/EDF identifier.

    The EDF file name must satisfy EyeLink constraints.
    This script uses the same value as participant ID for subset lookup.
    """
    allowed_char = ascii_letters + digits + '_'

    while True:
        dlg = gui.Dlg(title="Eye-tracking session")
        dlg.addText("Participant information")
        dlg.addText("Enter the participant ID used in the tables.")
        dlg.addField("Participant ID", initial="0")
        result = dlg.show()

        if not dlg.OK:
            print('user cancelled')
            core.quit()
            sys.exit()

        participant_id = str(result[0]).rstrip().split('.')[0]
        print(f"Participant ID: {participant_id}")

        if not participant_id:
            print('ERROR: Participant ID cannot be empty')
            continue
        if not all(c in allowed_char for c in participant_id):
            print('ERROR: Invalid EDF filename')
            continue
        if len(participant_id) > 8:
            print('ERROR: EDF filename should not exceed 8 characters')
            continue
        return participant_id


def normalize_modality(value):
    """Normalize table modality values to RGB or IR."""
    if pd.isna(value):
        return None

    value_str = str(value).strip().upper()
    if value_str in {"RGB", "VISIBLE"}:
        return "RGB"
    if value_str in {"IR", "infrared_"}:
        return "IR"

    raise RuntimeError(
        f"Unsupported modality value '{value}'. Expected RGB or IR "
        "(or visible / infrared_)."
    )


def modality_to_folder(modality):
    """Convert logical modality to the corresponding image/results folder."""
    modality = normalize_modality(modality)
    return MODALITY_TO_IMAGE_FOLDER[modality]


def prepare_results_folder(edf_name):
    """Create the session folder used to store session outputs."""
    global session_folder, session_identifier

    if current_modality_folder_global is None:
        raise RuntimeError("Modality folder is undefined. Load trials before preparing results.")

    results_root = RESULTS_BASE_DIR / current_modality_folder_global
    results_root.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
    session_identifier = edf_name + time_str
    session_folder = results_root / session_identifier
    session_folder.mkdir(parents=True, exist_ok=True)


def raise_if_missing(path, label):
    """Fail early with an explicit error if a required file is missing."""
    if not Path(path).exists():
        raise RuntimeError(f"{label} not found: {path}")


def compute_participant_seed(participant_id, subset):
    """Build a deterministic seed so each participant gets a stable order."""
    seed_str = f"{subset}::{participant_id}"
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(seed_str))


def shuffle_trials_for_participant(trials, participant_id, subset):
    """Shuffle trials reproducibly within a subset across participants."""
    shuffled_trials = list(trials)
    rng = random.Random(compute_participant_seed(participant_id, subset))
    rng.shuffle(shuffled_trials)
    return shuffled_trials


def format_local_edf_name(participant_id):
    """Build a local EDF filename compatible with Extractor's ID-(\d+) pattern."""
    match = re.search(r"(\d+)", str(participant_id))
    if match is None:
        safe_id = participant_id
    else:
        safe_id = f"{int(match.group(1)):03d}"

    timestamp = time.strftime("%Y_%m_%d_%H_%M", time.localtime())
    return f"ID-{safe_id}-{timestamp}.EDF"


def get_row_value(row, candidate_columns, default=None):
    """Return the first existing non-null value among several possible columns."""
    for column in candidate_columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


# -----------------------------------------------------------------------------
# EyeLink + PsychoPy initialization
# -----------------------------------------------------------------------------

def connect_tracker():
    """Connect to the EyeLink host, or run in dummy mode."""
    if DUMMY_MODE:
        return pylink.EyeLink(None)

    try:
        return pylink.EyeLink(HOST_IP)
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()


def open_edf_file(el_tracker, edf_name):
    """Open an EDF file on the Host PC and store its name globally."""
    global edf_file

    edf_file = edf_name + '.EDF'
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()

    preamble_text = 'RECORDED BY %s' % os.path.basename(__file__)
    el_tracker.sendCommand("add_file_preamble_text '%s'" % preamble_text)


def configure_tracker(el_tracker):
    """Configure file/link filters and core tracking parameters."""
    el_tracker.setOfflineMode()

    eyelink_ver = 0
    if not DUMMY_MODE:
        version_string = el_tracker.getTrackerVersionString()
        eyelink_ver = int(version_string.split()[-1].split('.')[0])
        print('Running experiment on %s, version %d' % (version_string, eyelink_ver))

    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'

    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'

    el_tracker.sendCommand(f"file_event_filter = {file_event_flags}")
    el_tracker.sendCommand(f"file_sample_data = {file_sample_flags}")
    el_tracker.sendCommand(f"link_event_filter = {link_event_flags}")
    el_tracker.sendCommand(f"link_sample_data = {link_sample_flags}")
    el_tracker.sendCommand("calibration_type = HV9")
    # el_tracker.sendCommand("validation_area_proportion 0.8 0.8")
    el_tracker.sendCommand("button_function 5 'accept_target_fixation'")


def create_window_and_graphics(el_tracker):
    """Create the PsychoPy window and calibration graphics environment."""
    global win, genv, screen_width, screen_height, stimulus_max_width, stimulus_max_height

    mon = monitors.Monitor(MONITOR_NAME, width=MONITOR_WIDTH_CM, distance=MONITOR_DISTANCE_CM)
    win = visual.Window(fullscr=FULL_SCREEN, monitor=mon, winType='pyglet', units='pix')

    screen_width, screen_height = win.size
    if 'Darwin' in platform.system() and use_retina:
        screen_width = int(screen_width / 2.0)
        screen_height = int(screen_height / 2.0)

    if STIMULUS_MAX_SIZE is None:
        stimulus_max_width, stimulus_max_height = screen_width, screen_height
    else:
        requested_width, requested_height = STIMULUS_MAX_SIZE
        stimulus_max_width = min(requested_width, screen_width)
        stimulus_max_height = min(requested_height, screen_height)

    el_tracker.sendCommand(
        "screen_pixel_coords = 0 0 %d %d" % (screen_width - 1, screen_height - 1)
    )
    el_tracker.sendMessage(
        "DISPLAY_COORDS  0 0 %d %d" % (screen_width - 1, screen_height - 1)
    )

    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv)

    foreground_color = (-1, -1, -1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)
    genv.setTargetType('picture')
    genv.setPictureTarget(str(CALIBRATION_TARGET))
    genv.setCalibrationSounds('', '', '')

    if use_retina:
        genv.fixMacRetinaDisplay()

    pylink.openGraphicsEx(genv)


# -----------------------------------------------------------------------------
# Geometry and stimulus preparation
# -----------------------------------------------------------------------------

def compute_image_display_geometry(image_path, max_width, max_height):
    """Compute displayed image size and rectangle without aspect distortion."""
    with Image.open(image_path) as im:
        src_width, src_height = im.size

    if src_width <= 0 or src_height <= 0:
        raise RuntimeError(f"Invalid image size for stimulus: {image_path}")

    scale = min(float(max_width) / float(src_width), float(max_height) / float(src_height))
    img_width = max(1, int(round(src_width * scale)))
    img_height = max(1, int(round(src_height * scale)))

    center_x = int(screen_width / 2.0)
    center_y = int(screen_height / 2.0)
    left = int(round((screen_width - img_width) / 2.0))
    top = int(round((screen_height - img_height) / 2.0))
    right = left + img_width - 1
    bottom = top + img_height - 1

    return Geometry(
        img_width=img_width,
        img_height=img_height,
        center_x=center_x,
        center_y=center_y,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )


def create_image_stimulus(image_path, geometry):
    """Create the PsychoPy stimulus with the already computed display size."""
    return visual.ImageStim(
        win,
        image=image_path,
        size=(geometry.img_width, geometry.img_height),
        pos=(0, 0),
    )


def send_host_image_backdrop(image_path, geometry):
    """Send the displayed image to the Host PC for Data Viewer support."""
    with Image.open(image_path) as im:
        im = im.resize((geometry.img_width, geometry.img_height))
        img_pixels = im.load()
        pixels = [
            [img_pixels[i, j] for i in range(geometry.img_width)]
            for j in range(geometry.img_height)
        ]

    el_tracker = pylink.getEYELINK()
    el_tracker.bitmapBackdrop(
        geometry.img_width,
        geometry.img_height,
        pixels,
        0,
        0,
        geometry.img_width,
        geometry.img_height,
        geometry.left,
        geometry.top,
        pylink.BX_MAXCONTRAST,
    )


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------

def clear_screen():
    """Clear the presentation window using the calibration background color."""
    win.fillColor = genv.getBackgroundColor()
    win.flip()


def show_message(text, wait_for_keypress=True):
    """Display an instruction or status message."""
    msg = visual.TextStim(
        win,
        text,
        color=genv.getForegroundColor(),
        wrapWidth=screen_width / 2,
    )
    clear_screen()
    msg.draw()
    win.flip()

    if wait_for_keypress:
        event.waitKeys()
        clear_screen()


# -----------------------------------------------------------------------------
# Trial loading and validation
# -----------------------------------------------------------------------------

def load_trials_for_participant(participant_id):
    """Load the participant subset and build parser-compatible trials.

    The loader is flexible regarding column names so the PsychoPy script can
    stay compatible with existing tables. The displayed file and the raw image
    name sent to the parser may be different.
    """
    global current_subset_global, current_modality_global, current_modality_folder_global

    raise_if_missing(PARTICIPANTS_TABLE, 'Participants table')
    raise_if_missing(STIMULI_TABLE, 'Images table')
    raise_if_missing(STIMULI_ROOT_DIR, 'Images directory')

    participants = pd.read_csv(PARTICIPANTS_TABLE)
    participant_match = participants.loc[
        participants['id'].astype(str) == str(participant_id),
        'subset',
    ]
    if participant_match.empty:
        raise RuntimeError(f"Participant ID '{participant_id}' not found in {PARTICIPANTS_TABLE}")
    subset = participant_match.iloc[0]
    current_subset_global = subset

    images_table = pd.read_csv(STIMULI_TABLE)
    subset_images = images_table[images_table['subset'] == subset]
    if subset_images.empty:
        raise RuntimeError(f"No images found for subset '{subset}' in {STIMULI_TABLE}")

    if 'modality' not in subset_images.columns:
        raise RuntimeError("The images table must contain a 'modality' column.")

    subset_modalities = subset_images['modality'].dropna().map(normalize_modality).unique()
    if len(subset_modalities) != 1:
        raise RuntimeError(
            f"Subset '{subset}' contains multiple modalities: {list(subset_modalities)}. "
            "Each subset must contain a single modality."
        )

    current_modality_global = subset_modalities[0]
    current_modality_folder_global = modality_to_folder(current_modality_global)

    trials = []
    for num_image, (_, row) in enumerate(subset_images.iterrows(), start=1):
        condition = get_row_value(row, ['label', 'condition'], default='NA')
        display_image = get_row_value(row, ['display_image', 'image', 'stimulus_image'])
        parser_image_name = get_row_value(
            row,
            ['parser_image_name', 'raw_image_name', 'image_name', 'image'],
        )
        row_modality = normalize_modality(row['modality'])
        image_folder = modality_to_folder(row_modality)

        if display_image is None:
            raise RuntimeError(
                f"No display image column found for subset '{subset}'. "
                "Expected one of: display_image, image, stimulus_image"
            )
        if parser_image_name is None:
            raise RuntimeError(
                f"No parser image name column found for subset '{subset}'. "
                "Expected one of: parser_image_name, raw_image_name, image_name, image"
            )

        image_path = STIMULI_ROOT_DIR / image_folder / str(display_image)
        raise_if_missing(image_path, 'Stimulus image')

        trials.append(
            Trial(
                condition=str(condition),
                display_image=str(display_image),
                parser_image_name=str(parser_image_name),
                num_image=num_image,
                display_image_path=str(image_path),
                display_image_relpath=(Path(image_folder) / str(display_image)).as_posix(),
            )
        )

    return shuffle_trials_for_participant(trials, participant_id, subset)


# -----------------------------------------------------------------------------
# Trial control
# -----------------------------------------------------------------------------

def abort_trial():
    """Stop the current recording and mark the trial as an error."""
    el_tracker = pylink.getEYELINK()

    if el_tracker.isRecording():
        pylink.pumpDelay(100)
        el_tracker.stopRecording()

    clear_screen()
    el_tracker.sendMessage('!V CLEAR 116 116 116')
    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_ERROR)
    return pylink.TRIAL_ERROR


def terminate_task():
    """Close recording, download EDF locally, and exit cleanly."""
    el_tracker = pylink.getEYELINK()

    if el_tracker.isConnected():
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            abort_trial()

        el_tracker.setOfflineMode()
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)
        el_tracker.closeDataFile()

        show_message('EDF data is transferring from EyeLink Host PC...', wait_for_keypress=False)
        local_edf = session_folder / format_local_edf_name(participant_id_global)
        try:
            el_tracker.receiveDataFile(edf_file, str(local_edf))
        except RuntimeError as error:
            print('ERROR:', error)

        el_tracker.close()

    if win is not None:
        win.close()

    core.quit()
    sys.exit()


def perform_drift_check(el_tracker):
    """Run drift correction before each trial."""
    while not DUMMY_MODE:
        if (not el_tracker.isConnected()) or el_tracker.breakPressed():
            terminate_task()
            return pylink.ABORT_EXPT

        try:
            error = el_tracker.doDriftCorrect(
                int(screen_width / 2.0),
                int(screen_height / 2.0),
                1,
                1,
            )
            if error is not pylink.ESC_KEY:
                break
        except Exception:
            pass
    return pylink.TRIAL_OK


def draw_fixation_cross():
    """Draw the central fixation cross before each stimulus."""
    fixation = visual.TextStim(win, text='+', color='black')
    clear_screen()
    fixation.draw()
    win.flip()


def send_trial_messages_before_stimulus(el_tracker, trial_index, trial, geometry):
    """Send trial start and parser/Data Viewer metadata to EyeLink.

    The order of messages matters for the current parser:
    - NUM_IMAGE ...
    - Affichage_Image
    - raw image name on the next MSG line
    """
    el_tracker.sendMessage('TRIALID %d' % trial_index)
    el_tracker.sendCommand("record_status_message 'TRIAL number %d'" % trial_index)

    # Parser-specific messages.
    el_tracker.sendMessage(f'NUM_IMAGE {trial.num_image}')
    el_tracker.sendMessage('Affichage_Image')
    el_tracker.sendMessage('!V TRIAL_VAR name %s' % trial.parser_image_name)
    el_tracker.sendMessage('!V TRIAL_VAR modality %s' % current_modality_folder_global)
    el_tracker.sendMessage('!V TRIAL_VAR stimulus_width %d' % geometry.img_width)
    el_tracker.sendMessage('!V TRIAL_VAR stimulus_height %d' % geometry.img_height)

    # Data Viewer support.
    el_tracker.sendMessage('!V CLEAR 116 116 116')
    bg_image = f"../../{STIMULI_ROOT_DIR.as_posix()}/{trial.display_image_relpath}"
    el_tracker.sendMessage(
        '!V IMGLOAD CENTER %s %d %d %d %d' % (
            bg_image,
            geometry.center_x,
            geometry.center_y,
            geometry.img_width,
            geometry.img_height,
        )
    )
    el_tracker.sendMessage(
        '!V IAREA RECTANGLE %d %d %d %d %d %s' % (
            1,
            geometry.left,
            geometry.top,
            geometry.right,
            geometry.bottom,
            'displayed_image',
        )
    )


def monitor_trial_keyboard(el_tracker, kb):
    """Check abort/terminate keyboard shortcuts during image display."""
    key_presses = kb.getKeys(keyList=None, waitRelease=False, clear=False)
    if not key_presses:
        return None

    names = [key.name for key in key_presses]

    if 'escape' in names:
        el_tracker.sendMessage('trial_skipped_by_user')
        clear_screen()
        abort_trial()
        return pylink.SKIP_TRIAL

    if 'c' in names and ('lctrl' in names or 'rctrl' in names):
        el_tracker.sendMessage('terminated_by_user')
        terminate_task()
        return pylink.ABORT_EXPT

    return None


def run_trial(trial, trial_index):
    """Run one full trial: fixation, image display, recording, cleanup."""
    image_path = trial.display_image_path

    geometry = compute_image_display_geometry(
        image_path=image_path,
        max_width=stimulus_max_width,
        max_height=stimulus_max_height,
    )
    image_stim = create_image_stimulus(image_path, geometry)

    el_tracker = pylink.getEYELINK()
    kb = keyboard.Keyboard()

    el_tracker.setOfflineMode()
    el_tracker.sendCommand('clear_screen 0')
    send_host_image_backdrop(image_path, geometry)

    el_tracker.setOfflineMode()
    try:
        el_tracker.startRecording(1, 1, 1, 1)
    except RuntimeError as error:
        print('ERROR:', error)
        abort_trial()
        return pylink.TRIAL_ERROR

    pylink.pumpDelay(100)

    draw_fixation_cross()
    core.wait(random.uniform(FIXATION_MIN_TIME, FIXATION_MAX_TIME))

    image_stim.draw()
    win.flip()
    image_onset_time = core.getTime()
    el_tracker.sendMessage('image_onset')

    send_trial_messages_before_stimulus(el_tracker, trial_index, trial, geometry)

    while True:
        if core.getTime() - image_onset_time >= IMG_TIME:
            el_tracker.sendMessage('time_out')
            break

        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            abort_trial()
            return error

        keyboard_status = monitor_trial_keyboard(el_tracker, kb)
        if keyboard_status is not None:
            return keyboard_status

    clear_screen()
    el_tracker.sendMessage('blank_screen')
    el_tracker.sendMessage('!V CLEAR 128 128 128')

    pylink.pumpDelay(100)
    el_tracker.stopRecording()

    el_tracker.sendMessage('TRIAL_RESULT %d' % pylink.TRIAL_OK)
    return pylink.TRIAL_OK


# -----------------------------------------------------------------------------
# Experiment flow
# -----------------------------------------------------------------------------

def show_start_message_and_calibrate(el_tracker):
    """Show initial instructions and open tracker setup if needed."""
    task_msg = 'In the task, you may Ctrl-C to if you need to quit the task early\n'
    if DUMMY_MODE:
        task_msg += '\nNow, press ENTER to start the task'
    else:
        task_msg += '\nNow, press ENTER twice to calibrate tracker'

    show_message(task_msg)

    if not DUMMY_MODE:
        try:
            el_tracker.doTrackerSetup()
        except RuntimeError as err:
            print('ERROR:', err)
            el_tracker.exitCalibration()


def run_central_recalibration(el_tracker, trial_index):
    """Run an explicit central drift correction block every N images."""
    if DUMMY_MODE:
        return

    clear_screen()
    perform_drift_check(el_tracker)


def run_full_recalibration(el_tracker, trial_index):
    """Run a full EyeLink setup/calibration block every N images."""
    if DUMMY_MODE:
        return

    clear_screen()
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration()


def apply_periodic_recalibration(el_tracker, trial_index):
    """Apply scheduled recalibration without changing the main trial logic."""
    if FULL_RECALIBRATION_EVERY and trial_index > 1 and (trial_index - 1) % FULL_RECALIBRATION_EVERY == 0:
        run_full_recalibration(el_tracker, trial_index)
    elif CENTRAL_RECALIBRATION_EVERY and trial_index > 1 and (trial_index - 1) % CENTRAL_RECALIBRATION_EVERY == 0:
        run_central_recalibration(el_tracker, trial_index)


def run_experiment_trials(el_tracker, trials):
    """Run all trials in order, with scheduled recalibration blocks."""
    for trial_index, trial in enumerate(trials, start=1):
        apply_periodic_recalibration(el_tracker, trial_index)
        run_trial(trial, trial_index)


def main():
    """Entry point for the full experiment."""
    switch_to_script_folder()
    ask_retina_mode_if_needed()

    global participant_id_global

    participant_id = ask_participant_and_edf_name()
    participant_id_global = participant_id

    raise_if_missing(CALIBRATION_TARGET, 'Calibration target')

    trials = load_trials_for_participant(participant_id)
    prepare_results_folder(participant_id)

    el_tracker = connect_tracker()
    open_edf_file(el_tracker, participant_id)
    configure_tracker(el_tracker)
    create_window_and_graphics(el_tracker)

    show_start_message_and_calibrate(el_tracker)
    run_experiment_trials(el_tracker, trials)
    terminate_task()


if __name__ == '__main__':

    main()
