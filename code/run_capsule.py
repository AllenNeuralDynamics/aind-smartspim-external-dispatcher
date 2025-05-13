""" Main script that works as a dispatcher in code ocean """

import json
import logging
import os
import re
import shutil
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from utils import utils

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler("test.log", "a"),
    ],
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PathLike = Union[str, Path]

SCRIPT_DIR = Path(os.path.abspath(__file__)).parent

PIPELINE_REPOS = [
    ("aind-smartspim-microscope-to-zarr", "File format conversion"),
    ("aind-smartspim-flatfield-estimation", "Image flat-field correction"),
    ("aind-smartspim-destripe", "Image destriping"),
    ("aind-smartspim-stitch", "Image tile alignment"),
    ("aind-smartspim-fuse", "Image tile fusing"),
    ("aind-smartspim-ccf-registration", "Image atlas alignment"),
    ("aind-smartspim-segmentation", "Image cell segmentation"),
    ("aind-smartspim-classification", "Image cell segmentation"),
    ("aind-smartspim-quantification", "Image cell quantification"),
]

MANIFEST_STEP_NAMES = {
    "stitching": {"possible_names": ["stitching"]},
    "registration": {"possible_names": ["registration", "ccf_registration"]},
    "segmentation": {"possible_names": ["segmentation", "cell_segmentation_channels"]},
}


def get_processing_manifest_path(raw_data_folder: str) -> str:
    """
    Gets the processing manifest path. It is necessary
    since the processing manifest is in different paths
    depending on the SmartSPIM version.

    Parameters
    ----------
    raw_data_folder: str
        Path of the raw data folder to perform
        the recursive search.

    Returns
    -------
    str
        Path where the processing manifest is stored.

    """
    raw_data_folder = Path(raw_data_folder)

    if not raw_data_folder.exists():
        raise FileNotFoundError(f"Raw data folder does not exist: {raw_data_folder}")

    processing_manifest_path = None

    for path in [
        raw_data_folder.joinpath("derivatives"),
        raw_data_folder.joinpath("SPIM/derivatives"),
    ]:
        curr_proc_man = path.joinpath("processing_manifest.json")
        if curr_proc_man.exists():
            processing_manifest_path = curr_proc_man

    return processing_manifest_path


def get_version(
    owner: str, repo: str, path: str, branch: Optional[str] = "main"
) -> str:
    """
    Gets the version of a repository,

    Parameters
    ----------
    owner: str
        Github owner of the repository.

    repo: str
        Repository name

    path: str
        Path within the repository

    branch: Optional[str]
        Branch from where we will pull
        the version. Default: "main"

    Returns
    -------
    str
        String with the version
    """
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    response = requests.get(url)

    if response.status_code == 200:
        match = re.search(r'__version__\s*=\s*"([^"]+)"', response.text)
        return match.group(1) if match else "Version not found"
    else:
        return None


def get_pipeline_versions(
    pipeline_repos: List, owner: Optional[str] = "AllenNeuralDynamics"
) -> Dict:
    """
    Gets the SmartSPIM pipeline version for
    each of the image processing steps.

    Parameters
    ----------
    pipeline_repos: List
        List with tuples correspoding to Tuple[
            repo_name, metadata name in aind schema
        ]

    owner: Optional[str]
        Repository owner.
        Default: "AllenNeuralDynamics"

    Returns
    -------
    Dict
        Dictionary with the versions of the latest
        version of each of the SmartSPIM pipeline steps.
    """
    step_versions = {}

    for repo, step_name in pipeline_repos:
        if "ccf" in repo:
            package_name = "aind_ccf_reg"
        else:
            package_name = repo.replace("-", "_")

        version = get_version(owner, repo, path=f"code/{package_name}/__init__.py")
        step_versions[f"{repo} - {step_name}"] = {
            "version": version,
        }

    return step_versions


def get_dataset_step_versions(dataset_path: str) -> Dict:
    """
    Gets the dataset step versions from the processing.json

    Parameters
    ----------
    dataset_path: str
        Path to the dataset folder

    Returns
    -------
    Dict
        Dictionary with the versions of the image processing steps
        in the SmartSPIM pipeline for a given dataset.
    """
    processing_path = Path(dataset_path).joinpath("processing.json")
    dataset_step_versions = None

    if processing_path.exists():
        try:
            processing_data = utils.read_json_as_dict(filepath=str(processing_path))
        except BaseException as e:
            print(f"Error reading {processing_path}: {e}")
            processing_data = {}

        processing_pipeline = processing_data.get("processing_pipeline")
        pipeline_steps = processing_data.get("data_processes")

        if pipeline_steps is None:
            pipeline_steps = (
                processing_pipeline.get("data_processes")
                if processing_pipeline
                else None
            )

        if pipeline_steps:
            dataset_step_versions = {}

            for step in pipeline_steps:
                code_url = step.get("code_url")
                step_name = step.get("name")
                code_version = step.get("software_version", step.get("version"))

                package_name = code_url.split("/")[-1]
                dataset_step_versions[f"{package_name} - {step_name}"] = {
                    "version": code_version
                }

        else:
            print(f"No pipeline steps found in {processing_path}: {processing_data}")

    else:
        print("PROCESSING PATH DOES NOT EXIST: ", dataset_path.stem, processing_path)

    return dataset_step_versions


def check_dataset_latest_version(
    dataset_versions: Dict,
    latest_versions: Dict,
):
    """
    Checks within the metadata (processing.json)
    and the image processing versions to see
    if any of the steps need to be rerun. If it
    is not the latest version, the step will
    be flagged as True to reprocess.

    Parameters
    ----------
    dataset_versions: Dict
        Image processing steps with their versions
        in the SmartSPIM pipeline for a given dataset.
        This metadata can be found in the processing.json

    latest_versions: Dict
        Latest versions of the image processing steps
        in the SmartSPIM pipeline. This is related to
        the pipeline and not a specific dataset.

    Returns
    -------
    Dict
        Dictionary for each of the steps that dictates
        if we need to process a specific step in the pipeline.
    """
    process_versions = {}

    for step, values in latest_versions.items():
        dataset_step = dataset_versions.get(step)

        if dataset_step is None:
            curr_key = None
            if "tile alignment" in step:
                curr_key = [
                    d
                    for d in list(dataset_versions.keys())
                    if "tile alignment" in d.lower()
                ]

            elif "tile fusing" in step:
                curr_key = [
                    d
                    for d in list(dataset_versions.keys())
                    if "tile fusing" in d.lower()
                ]

            elif "atlas alignment" in step:
                curr_key = [
                    d
                    for d in list(dataset_versions.keys())
                    if "atlas alignment".lower() in d.lower()
                ]

            curr_key = curr_key[0] if curr_key and len(curr_key) else None
            dataset_step = dataset_versions.get(curr_key)

        values_version = values.get("version")

        process_versions[step] = {
            "process": True,
            "latest_version": values_version,
            "dataset_version": None,
        }

        if dataset_step:
            dataset_step_version = dataset_step.get("version")
            if dataset_step_version == values_version:
                process_versions[step] = {
                    "process": False,
                    "dataset_version": dataset_step_version,
                    "latest_version": values_version,
                }

            else:
                process_versions[step]["dataset_version"] = dataset_step_version

    return process_versions


def get_standard_manifest_config(pipeline_processing: Dict, hashmap_stepnames: Dict):
    """
    Reads a processing manifest configuration
    and converts it to the standard.

    Parameters
    ----------
    pipeline_processing: Dict
        Pipeline processing manifest.
        This could be from a very old version.

    hashmap_stepnames: Dict
        Hashmap with the step names

    Parameters
    ----------
    Dict
        Dictionary with the new processing manifest.
    """

    if not len(pipeline_processing):
        raise ValueError("Please, provide a valid processing manifest.")

    standard_pipeline_processing = {}
    for step_name, values in hashmap_stepnames.items():
        standard_pipeline_processing[step_name] = {}

        for possible_name in values["possible_names"]:
            if possible_name in pipeline_processing:
                config = {}
                if "cell_segmentation_channels" == possible_name:
                    # Cell finder's params
                    config = {
                        "channels": pipeline_processing[possible_name],
                        "input_scale": "0",
                        "chunksize": "128",
                        "signal_start": "0",
                        "signal_end": "-1",
                    }

                elif "ccf_registration" == possible_name:
                    config = {
                        "channels": pipeline_processing[possible_name],
                        "input_scale": 3,
                    }

                else:
                    config = pipeline_processing[possible_name]

                standard_pipeline_processing[step_name] = config
                break

    return standard_pipeline_processing


def get_omezarr_path(stitched_path: str) -> str:
    """
    Gets the path where the fused data
    is stored. It is necessary since the
    fused data could be in different folders
    depending the SmartSPIM folder structure
    version.

    Parameters
    ----------
    stitched_path: str
        Root path of the stitched data asset.

    Returns
    -------
    str
        Path where the OMEZarrs are stored.
    """
    stitched_path = Path(stitched_path)

    if not stitched_path.exists():
        raise FileNotFoundError(f"Path {stitched_path} does not exist!")

    possible_omezarr_folders = ["processed", "image_tile_fusing"]

    for pof in possible_omezarr_folders:
        curr_folder = stitched_path.joinpath(pof)

        if curr_folder.joinpath("OMEZarr").exists():
            return curr_folder

    return None


def wavelength_to_hex(wavelength: int) -> int:
    """
    Converts wavelength to corresponding color hex value.

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """
    # Each wavelength key is the upper bound to a wavelgnth band.
    # Wavelengths range from 380-750nm.
    # Color map wavelength/hex pairs are generated by sampling
    # along a CIE diagram arc.

    color_map = {
        460: 0x690AFE,  # Purple
        470: 0x3F2EFE,  # Blue-Purple
        480: 0x4B90FE,  # Blue
        490: 0x59D5F8,  # Blue-Green
        500: 0x5DF8D6,  # Green
        520: 0x5AFEB8,  # Green
        540: 0x58FEA1,  # Green
        560: 0x51FF1E,  # Green
        565: 0xBBFB01,  # Green-Yellow
        575: 0xE9EC02,  # Yellow
        580: 0xF5C503,  # Yellow-Orange
        590: 0xF39107,  # Orange
        600: 0xF15211,  # Orange-Red
        620: 0xF0121E,  # Red
        750: 0xF00050,
    }  # Pink

    for ub, hex_val in color_map.items():
        if wavelength < ub:  # Exclusive
            return hex_val
    return hex_val  # hex_val is set to the last color in for loop


def str_to_bool(s: str):
    """
    Parsing string to boolean

    Parameters
    ----------
    s: str
        String to parse. Admitted values true or false.

    Raises
    ------
    ValueError
        If the string is not true or false.

    Returns
    -------
    bool
        Parsed string to boolean

    """
    s_cleaned = s.strip().lower().replace("'", "")
    if s_cleaned == "true":
        return True
    elif s_cleaned == "false":
        return False
    else:
        raise ValueError(f"Input should be 'true' or 'false'. Provided: {s_cleaned}")


def get_dataset_post_processing_config(
    processed_step_versions: Dict, pipeline_processing: Dict, latest_step_versions: Dict
):
    """
    Creates the configuration for the postprocessing pipeline.
    The idea is that if the versions of the image processing
    steps is different, then it will have to be executed.

    However, a step will only be executed if it has a
    configuration within the processing manifest.

    Parameters
    ----------
    processed_step_version: Dict
        Versions of the image processing steps that were
        executed for the dataset.

    pipeline_processing: Dict
        Dictionary with the steps that need to be executed
        for this dataset. It is the configuration within
        the processing_manifest.json in derivatives.

    latest_step_version: Dict
        Dictionary with the latest versions of the
        image processing steps published in the pipeline.

    Returns
    -------
    Dict
        Dictionary with the final configuration
        for each of the image processing steps.
    """
    final_config = {
        "pipeline_processing": pipeline_processing,
        "need_registration": {},
        "need_proposals": {},
        "need_classification": {},
        "need_quantification": {},
    }

    if processed_step_versions and pipeline_processing:
        process_versions = check_dataset_latest_version(
            processed_step_versions, latest_step_versions
        )

        image_reg_cfg = pipeline_processing.get("registration")
        image_seg_cfg = pipeline_processing.get("segmentation")

        reg_channels = image_reg_cfg.get("channels")
        seg_channels = image_seg_cfg.get("channels")

        version_control_reg = process_versions[
            "aind-smartspim-ccf-registration - Image atlas alignment"
        ]["process"]

        # Checking if there's something in the manifest
        manifest_reg = reg_channels[0] if reg_channels and len(reg_channels) else []
        manifest_seg = seg_channels[0] if seg_channels and len(seg_channels) else []

        version_control_proposals = process_versions[
            "aind-smartspim-segmentation - Image cell segmentation"
        ]["process"]

        version_control_classification = process_versions[
            "aind-smartspim-classification - Image cell segmentation"
        ]["process"]

        version_control_quantification = process_versions[
            "aind-smartspim-quantification - Image cell quantification"
        ]["process"]

        if len(manifest_reg) and version_control_reg:
            need_reg = process_versions[
                "aind-smartspim-ccf-registration - Image atlas alignment"
            ]

        if len(manifest_seg):
            # Might need segmentation, classification or quantification
            if version_control_proposals:
                final_config["need_proposals"] = process_versions[
                    "aind-smartspim-segmentation - Image cell segmentation"
                ]
                final_config["need_classification"] = process_versions[
                    "aind-smartspim-classification - Image cell segmentation"
                ]
                final_config["need_quantification"] = process_versions[
                    "aind-smartspim-quantification - Image cell quantification"
                ]

            elif version_control_classification:
                final_config["need_classification"] = process_versions[
                    "aind-smartspim-classification - Image cell segmentation"
                ]
                final_config["need_quantification"] = process_versions[
                    "aind-smartspim-quantification - Image cell quantification"
                ]

            elif version_control_quantification or len(need_reg):
                final_config["need_quantification"] = process_versions[
                    "aind-smartspim-quantification - Image cell quantification"
                ]

    elif pipeline_processing:
        image_reg_cfg = pipeline_processing.get("registration")
        image_seg_cfg = pipeline_processing.get("segmentation")

        reg_channels = image_reg_cfg.get("channels")
        seg_channels = image_seg_cfg.get("channels")

        manifest_reg = reg_channels[0] if reg_channels and len(reg_channels) else []
        manifest_seg = seg_channels[0] if seg_channels and len(seg_channels) else []

        if len(manifest_reg):
            final_config["need_registration"] = {"process": True}

        # Trigger everything if processing.json does not exist
        if len(manifest_seg):
            # Might need segmentation, classification or quantification
            final_config["need_proposals"] = {"process": True}
            final_config["need_classification"] = {"process": True}
            final_config["need_quantification"] = {"process": True}

    else:
        print(
            f"[!!!] Problem getting the process versions: {process_versions} - manifest: {pipeline_processing}"
        )

    return final_config


def wavelength_to_hex_alternate(wavelength: int) -> int:
    """
    Converts wavelengths to hex value, taking fpbase.org spectra viewer
    as a guide.
    Fluorescent proteins querried:
    mTFP1,
    EGFP,
    SYFP2,
    mbanana,
    morange,
    mtomato,
    mcherry,
    mraspberry,
    mplum

    Parameters
    ------------------------
    wavelength: int
        Integer value representing wavelength.

    Returns
    ------------------------
    int:
        Hex value color.
    """

    color_map = {
        500: 0x61ABFD,  # RUDDY BLUE, mTFP/mTurquoise
        530: 0x92FF42,  # CHARTREUSE,   EGFP
        540: 0xE4FE41,  # CHARTREUSE, SYFP2
        560: 0xF3D038,  # MUSTARD, mBanana
        580: 0xEAB032,  # XANTHOUS, mOrange
        600: 0xF15F22,  # GIANTS ORANGE, tdTomato/mScarlet
        630: 0xED1C24,  # RED, mCherry
        680: 0xC51E1F,  # FIRE ENGINE RED, mRaspberry
        700: 0xA81F1F,  # FIRE BRICK, mPlum
    }

    for ub, hex_val in color_map.items():
        if wavelength <= ub:  # Inclusive
            return hex_val
    return hex_val  # hex_val is set to the last color in for loop


def volume_orientation(acquisition_params: dict):
    """
    Uses the acquisition orientation to set the cross-section
    orientation in the neuroglancer links

    Parameters
    ----------
    acquisition_params : dict
        acquisition paramenters from the processing manifest

    Raises
    ------
    ValueError
        if a brain is aquired in a way other than those predifined here

    Returns
    -------
    orientation : list
        orientation values for the neuroglancer link

    """

    acquired = ["", "", ""]

    for axis in acquisition_params["axes"]:
        acquired[axis["dimension"]] = axis["direction"][0]

    acquired = "".join(acquired)

    if acquired in ["SPR", "SPL"]:
        orientation = [0.5, 0.5, 0.5, -0.5]
    elif acquired == "SAL":
        orientation = [0.5, 0.5, -0.5, 0.5]
    elif acquired == "IAR":
        orientation = [0.5, -0.5, 0.5, 0.5]
    elif acquired == "RAS":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]
    elif acquired == "RPI":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, -np.cos(np.pi / 4)]
    elif acquired == "LAI":
        orientation = [0.0, np.cos(np.pi / 4), -np.cos(np.pi / 4), 0.0]
    else:
        raise ValueError(
            "Acquisition orientation: {acquired} has unknown NG parameters"
        )

    return orientation


def dispatch(processing_manifest: dict, results_folder: PathLike):
    """
    Creates multiple processing manifest jsons using
    the original processing manifest. This is done to
    use the flatten connection and instantiate multiple
    computations to process each channel in parallel.

    Parameters
    ----------
    processing_manifest: dict
        Dictionary with the processing manifest
        metadata

    results_folder: str
        Path pointing to the results folder

    """

    logger.info(f"Provided processing manifest: {processing_manifest}")

    # Creating processing manifests for channels to register
    pipeline_config = processing_manifest.get("pipeline_processing")

    if pipeline_config:
        logger.info("Creating segmentation and quantification parameters")
        # Creating processing manifests for channels to segment and quantify
        segment_channels = pipeline_config["segmentation"]["channels"]
        background_channel = processing_manifest["pipeline_processing"]["registration"][
            "channels"
        ][0]

        if len(segment_channels):
            print(f"Preparing segmentation configs for: {segment_channels}")

            for channel_to_segment in segment_channels:
                copy_pipeline_config = pipeline_config.copy()

                copy_pipeline_config["segmentation"]["input_data"] = "../data/fused"
                copy_pipeline_config["segmentation"]["channel"] = channel_to_segment
                copy_pipeline_config["segmentation"][
                    "background_channel"
                ] = background_channel

                # Creating quantification parameters
                copy_pipeline_config["quantification"] = {}
                copy_pipeline_config["quantification"]["fused_folder"] = "../data/fused"
                copy_pipeline_config["quantification"]["channel"] = channel_to_segment
                copy_pipeline_config["quantification"]["save_path"] = "../results/"

                utils.save_dict_as_json(
                    f"{results_folder}/segmentation_processing_manifest_{channel_to_segment}.json",
                    copy_pipeline_config,
                )

        else:
            utils.save_dict_as_json(
                f"{results_folder}/segmentation_processing_manifest_empty.json",
                pipeline_config.copy(),
            )

            print(
                f"No segmentation channels provided, pipeline config: {pipeline_config}"
            )

    else:
        raise BaseException("Stopping pipeline, pipeline configuration.")


def clean_up(
    processing_manifest: dict,
    data_folder: PathLike,
    results_folder: PathLike,
    cloud_mode: bool,
):
    """
    Moves all the data to the aind-open-data bucket in
    AWS.

    Parameters
    ----------
    processing_manifest: dict
        Dictionary with the processing manifest
        metadata

    data_folder: str
        Path pointing to the data folder

    results_folder: str
        Path pointing to the results folder

    cloud_mode: bool
        True if you're moving the data to the cloud,
        False otherwise.

    """
    logger.info(f"Data folder: {os.listdir(data_folder)}")

    # # Variables from processing manifest
    # bucket = "aind-open-data"

    cell_folders = glob(f"{data_folder}/cell_*")
    quantification_folders = glob(f"{data_folder}/quant_*")

    logger.info(f"Cell folders: {cell_folders}")
    logger.info(f"Quantification folders: {quantification_folders}")

    # Reading segmentation processings
    segmentation_processing = []
    for cell_folder in cell_folders:
        processing_jsons = [
            p
            for p in glob(f"{cell_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        segmentation_processing.append(processing_jsons)

    # Reading quantification processings
    quantification_processing = []
    for quant_folder in quantification_folders:
        processing_jsons = [
            p
            for p in glob(f"{quant_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        quantification_processing.append(processing_jsons)

    # Building from previous processing json
    processing_paths = list()
    combined_processing_list = (
        [[f"{data_folder}/output_aind_metadata/processing.json"]]
        + segmentation_processing
        + quantification_processing
    )
    for sub_list in combined_processing_list:
        processing_paths += sub_list

    logger.info(f"Compiling processing paths: {processing_paths}")

    if len(processing_paths) > 1:
        output_filename = utils.compile_processing_jsons(
            processing_paths=processing_paths,
            output_general_processing=results_folder,
            processor_full_name="Camilo Laiton",
            pipeline_version="3.0.1",
            pipeline_notes="SLURM pipeline",
        )
        logger.info(f"Compiled processing.json in path {output_filename}")

        # Moving data out
        if cloud_mode:
            # Defining s3 outputs
            s3_path = processing_manifest["pipeline_processing"]["stitching"]["s3_path"]
            cell_s3_output = f"{s3_path}/image_cell_segmentation"
            quantification_s3_output = f"{s3_path}/image_cell_quantification"

            regex_channels = r"Ex_(\d{3})_Em_(\d{3})$"

            # Copying final processing manifest
            for out in utils.execute_command_helper(
                f"aws s3 mv {results_folder}/processing.json {s3_path}/processing.json"
            ):
                print(out)

            # Moving data to the cell folder
            for cell_folder in cell_folders:
                channel_name = re.search(regex_channels, cell_folder).group()

                for out in utils.execute_command_helper(
                    f"aws s3 mv --recursive {cell_folder} {cell_s3_output}/{channel_name}"
                ):
                    print(out)

            # Moving data to the quantification folder
            for quantification_folder in quantification_folders:
                channel_name = re.search(regex_channels, quantification_folder).group()

                for out in utils.execute_command_helper(
                    f"aws s3 mv --recursive {quantification_folder} {quantification_s3_output}/{channel_name}"
                ):
                    print(out)

        else:
            # Move the data locally
            s3_path = processing_manifest["pipeline_processing"]["stitching"]["s3_path"]
            cell_s3_output = f"{s3_path}/image_cell_segmentation"
            quantification_s3_output = f"{s3_path}/image_cell_quantification"

            regex_channels = r"Ex_(\d{3})_Em_(\d{3})$"

            # Copying final processing manifest
            for out in utils.execute_command_helper(
                f"mv {results_folder}/processing.json {s3_path}/processing.json"
            ):
                print(out)

            # Moving data to the cell folder
            for cell_folder in cell_folders:
                channel_name = re.search(regex_channels, cell_folder).group()
                dest_folder = f"{cell_s3_output}/{channel_name}"
                utils.create_folder(dest_folder, verbose=True)

                for out in utils.execute_command_helper(
                    f"mv {cell_folder}/* {dest_folder}/"
                ):
                    print(out)

            # Moving data to the quantification folder
            for quantification_folder in quantification_folders:
                channel_name = re.search(regex_channels, quantification_folder).group()
                dest_folder = f"{quantification_s3_output}/{channel_name}"
                utils.create_folder(dest_folder, verbose=True)

                for out in utils.execute_command_helper(
                    f"mv {quantification_folder}/* {dest_folder}/"
                ):
                    print(out)

        utils.save_string_to_txt(
            f"Results of cell segmentation saved in: {cell_s3_output}",
            f"{results_folder}/output_cell.txt",
        )

        utils.save_string_to_txt(
            f"Results of quantification saved in: {quantification_s3_output}",
            f"{results_folder}/output_quantification.txt",
        )

    else:
        raise BaseException(f"Stopping clean up, no processing jsons found!")


def get_data_config(
    data_folder: PathLike,
    processing_manifest_path: str = "processing_manifest.json",
    data_description_path: str = "data_description.json",
) -> Tuple:
    """
    Returns the first smartspim dataset found
    in the data folder

    Parameters
    -----------
    data_folder: str
        Path to the folder that contains the data

    processing_manifest_path: str
        Path for the processing manifest

    data_description_path: str
        Path for the data description

    Returns
    -----------
    Tuple[Dict, str, list]
        Dict: Empty dictionary if the path does not exist,
        dictionary with the data otherwise.

        Str: Empty string if the processing manifest
        was not found

        List: Empty list if no investigators in data description
    """

    # Returning first smartspim dataset found
    # Doing this because of Code Ocean, ideally we would have
    # a single dataset in the pipeline

    processing_manifest_path = Path(f"{data_folder}/{processing_manifest_path}")
    data_description_path = Path(f"{data_folder}/{data_description_path}")

    if not processing_manifest_path.exists():
        raise ValueError(
            f"Please, check processing manifest path: {processing_manifest_path}"
        )

    if not data_description_path.exists():
        raise ValueError(
            f"Please, check data description path: {data_description_path}"
        )

    derivatives_dict = utils.read_json_as_dict(str(processing_manifest_path))
    data_description_dict = utils.read_json_as_dict(str(data_description_path))

    smartspim_dataset = data_description_dict["name"]
    investigators = data_description_dict["investigators"]

    return derivatives_dict, smartspim_dataset, investigators


def log_and_execute(cmd: str):
    """
    Logs the command and executes it

    Parameters
    ----------
    cmd: str
        Command to execute
    """
    logger.info(f"Executing CMD: {cmd}")
    for out in utils.execute_command_helper(cmd):
        logger.info(out)


def copy_to_s3(local_path: str, s3_dest: str, recursive: Optional[bool] = True):
    """
    Copies a local path to an S3 destination.
    Parameters
    ----------
    local_path: str
        Path to the local file or folder
    s3_dest: str
        Path to the S3 destination
    recursive: bool
        If True, the copy will be recursive
        (i.e. it will copy all files and folders
        within the local path). Default: True
    """
    flag = "--recursive" if recursive else ""
    cmd = f"aws s3 cp {flag} {local_path} {s3_dest}".strip()
    log_and_execute(cmd)


def move_to_s3(local_path: str, s3_dest: str):
    """
    Moves data from a local path to an S3 destination.

    Parameters
    ----------
    local_path: str
        Path to the local file or folder

    s3_dest: str
        Path to the S3 destination
    """
    cmd = f"aws s3 mv --recursive {local_path} {s3_dest}"
    log_and_execute(cmd)


def copy_intermediate_data(
    output_dispatch_metadata: PathLike,
    flatfield_folder: List[PathLike],
    destripe_files: List[PathLike],
    stitch_folder: List[PathLike],
    fuse_folder: List[PathLike],
    ccf_folders: List[PathLike],
    new_dataset_name: str,
    output_path: str,
    results_folder: PathLike,
    logger: logging.Logger,
    cloud_mode: bool = False,
):
    """
    Copies the destripe, stitch and fusion metadata
    to the destination bucket to make it available
    to scientists as soon as possible.

    Parameters
    ----------
    output_dispatch_metadata: PathLike
        Path where the new metadata (derived)
        for the processed dataset is located

    destripe_files: List[PathLike]
        Metadata files generated in the
        parallel destriping step

    flatfield_channels: List[PathLike]
        Flatfields applied to the dataset

    stitch_folders: List[PathLike]
        Stitch folders generated in the
        stitch step.

    fuse_folders: List[PathLike]
        Fuse folders generated in the
        parallel fusion step.

    ccf_folders: List[PathLike]
        CCF registration folders generated
        in the pipeline.

    s3_path: str
        Path where we want to copy the data to s3.

    results_folder: PathLike
        Results folder path in Code Ocean

    logger: logging.Logger
        Logging object

    """
    flatfield_processings = [str(flatfield_folder.joinpath("metadata/processing.json"))]
    stitch_processings = [str(stitch_folder.joinpath("metadata/processing.json"))]
    fuse_processings = [str(p) for p in list(fuse_folder.glob("*_processing.json"))]
    ccf_processings = []

    for ccf_folder in ccf_folders:
        processing_jsons = [
            p
            for p in glob(f"{ccf_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        ccf_processings.append(processing_jsons)

    # Flattening list
    processing_paths = list()
    combined_processing_list = ccf_processings
    for sub_list in combined_processing_list:
        processing_paths += sub_list

    processing_paths = (
        flatfield_processings
        + destripe_files
        + stitch_processings
        + fuse_processings
        + processing_paths
    )
    logger.info(f"Processing paths: {processing_paths}")

    try:
        output_filename = utils.compile_processing_jsons(
            processing_paths=processing_paths,
            output_general_processing=output_dispatch_metadata,
            processor_full_name="Camilo Laiton",
            pipeline_version="3.0.1",
            pipeline_notes="SLURM pipeline",
        )

    except Exception as e:
        print(f"Error while compiling processing manifests: {e}")
        output_filename = None

    logger.info(f"Compiled processing.json in path {output_filename}")

    if cloud_mode:
        s3_path = f"s3://{output_path}/{new_dataset_name}"
        output_dispatch_metadata = Path(output_dispatch_metadata)

        # Copying derived metadata
        copy_to_s3(output_dispatch_metadata, s3_path)

        # Fused data
        output_fusion = "image_tile_fusing"
        dest_metadata_path = f"{s3_path}/{output_fusion}/metadata"
        dest_zarr_path = f"{s3_path}/{output_fusion}/OMEZarr"

        copy_to_s3(flatfield_folder, f"{dest_metadata_path}/flatfield_correction")

        for fused_zarr in fuse_folder.glob("*.zarr"):
            fused_zarr_path = str(fused_zarr)
            fused_zarr_name = fused_zarr.name
            copy_to_s3(fused_zarr_path, f"{dest_zarr_path}/{fused_zarr_name}")

        fused_metadata_files = list(fuse_folder.glob("*.yaml")) + list(
            fuse_folder.glob("*.json")
        )
        for fused_metadata in fused_metadata_files:
            copy_to_s3(
                fused_metadata,
                f"{dest_metadata_path}/fusion/{fused_metadata.name}",
                recursive=False,
            )

        copy_to_s3(stitch_folder, f"{dest_metadata_path}/stitching")

        # CCF data
        ccf_s3_output = f"{s3_path}/image_atlas_alignment"
        regex_channels = r"Ex_(\d{3})_Em_(\d{3})|ccf_reverse|ccf_annotation_precomputed"

        for ccf_folder in ccf_folders:
            channel_name = re.search(regex_channels, ccf_folder).group()
            move_to_s3(ccf_folder, f"{ccf_s3_output}/{channel_name}")

    else:
        output_dispatch_metadata = Path(output_dispatch_metadata)
        local_path = Path(output_path) / new_dataset_name
        local_path.mkdir(parents=True, exist_ok=True)

        shutil.copytree(output_dispatch_metadata, local_path, dirs_exist_ok=True)
        logger.info(f"Copied metadata from {output_dispatch_metadata} to {local_path}")

        output_fusion = "image_tile_fusing"
        dest_metadata_path = local_path / output_fusion / "metadata"
        dest_zarr_path = local_path / output_fusion / "OMEZarr"

        dest_flatfield_path = dest_metadata_path / "flatfield_correction"
        shutil.copytree(flatfield_folder, dest_flatfield_path, dirs_exist_ok=True)
        logger.info(
            f"Copied flatfield data from {flatfield_folder} to {dest_flatfield_path}"
        )

        for fused_zarr in fuse_folder.glob("*.zarr"):
            dest_zarr = dest_zarr_path / fused_zarr.name
            shutil.copytree(fused_zarr, dest_zarr, dirs_exist_ok=True)
            logger.info(f"Copied data from {fused_zarr} to {dest_zarr}")

        fused_metadata_files = list(fuse_folder.glob("*.yaml")) + list(
            fuse_folder.glob("*.json")
        )
        fusion_metadata_path = dest_metadata_path / "fusion"
        fusion_metadata_path.mkdir(parents=True, exist_ok=True)

        for fused_metadata in fused_metadata_files:
            dest_file = fusion_metadata_path / fused_metadata.name
            shutil.copy2(fused_metadata, dest_file)
            logger.info(f"Copied data from {fused_metadata} to {dest_file}")

        stitch_dest_path = dest_metadata_path / "stitching"
        shutil.copytree(stitch_folder, stitch_dest_path, dirs_exist_ok=True)
        logger.info(f"Copied data from {stitch_folder} to {stitch_dest_path}")

        # CCF data
        ccf_output = local_path / "image_atlas_alignment"
        regex_channels = r"Ex_(\d{3})_Em_(\d{3})|ccf_reverse|ccf_annotation_precomputed"

        for ccf_folder in ccf_folders:
            ccf_folder_path = Path(ccf_folder)
            match = re.search(regex_channels, ccf_folder)
            if match:
                channel_name = match.group()
                dest_ccf_path = ccf_output / channel_name
                shutil.move(ccf_folder_path, dest_ccf_path)
                logger.info(f"Moved CCF folder {ccf_folder_path} to {dest_ccf_path}")
            else:
                logger.warning(f"No channel match found for {ccf_folder}")

        s3_path = str(local_path)
        dest_zarr_path = str(dest_zarr_path)

    utils.save_string_to_txt(
        f"Stitched dataset saved in: {local_path}",
        f"{results_folder}/output_stitching.txt",
    )

    return s3_path, dest_zarr_path


def create_derived_stitched_metadata(
    data_folder: PathLike, results_folder: PathLike, logger: logging.Logger
) -> Tuple[PathLike, str]:
    """
    Creates the derived metadata following
    AIND conventions.

    Parameters
    ----------
    data_folder: PathLike
        Path to the code ocean data folder

    results_folder: PathLike
        Path to the code ocean results folder

    logger: logging.Logger
        Logging object

    Returns
    -------
    Tuple[PathLike, str]
        The first position of the tuple
        corresponds to the path where the
        metadata was created while the
        second position has the new name
        of the dataset
    """
    logger.info("Generating derived data description")
    raw_metadata_path = data_folder.joinpath("input_aind_metadata")
    output_dispatch_metadata = f"{results_folder}/output_aind_metadata"
    utils.create_folder(output_dispatch_metadata)

    print(f"Contents raw metadata folder: {os.listdir(raw_metadata_path)}")
    print(f"Contents data folder: {os.listdir(data_folder)}")

    new_dataset_name = utils.generate_data_description(
        raw_data_description_path=raw_metadata_path.joinpath("data_description.json"),
        dest_data_description=output_dispatch_metadata,
        process_name="stitched",
    )

    logger.info("Copying all available raw SmartSPIM metadata")

    # This is the AIND metadata
    found_metadata = utils.copy_available_metadata(
        input_path=raw_metadata_path,
        output_path=output_dispatch_metadata,
        files_to_copy=[
            "acquisition.json",
            "instrument.json",
            "subject.json",
            "procedures.json",
            "session.json",
        ],
    )

    logger.info(f"Copied metadata from {raw_metadata_path}: {found_metadata}")
    logger.info(
        f"Metadata in raw folder {raw_metadata_path}: {os.listdir(raw_metadata_path)}"
    )
    logger.info(
        f"Metadata in folder {output_dispatch_metadata}: {os.listdir(output_dispatch_metadata)}"
    )

    return output_dispatch_metadata, new_dataset_name


def create_neuroglancer_link(
    config: dict,
    s3_channel_paths: List[str],
    s3_dataset_path: str,
    orientation: dict,
    dynamic_ranges: dict,
    segmentation: bool,
) -> str:
    """
    Creates the neuroglancer link for the processed dataset

    Parameters
    -------------

    config: dict
        Image configuration necessary to build the
        neuroglancer link

    s3_channel_paths: List[str]
        S3 paths for each of the channels

    s3_dataset_path: str
        S3 path where the dataset is stored

    orientation: dict
        Acquisition orientation obtained from processing manifest

    dynamic_ranges: dict
        Values for setting dynamic range for each channel

    Returns
    -------------
    Tuple[str, str]
        str:
            Path where the neuroglancer config json
            was generated
        str:
            Neuroglancer link path
    """
    # Sort channels paths so that they appear in NG consistently ordered
    s3_channel_paths = sorted(s3_channel_paths)

    dimensions = {
        "z": [
            config["z_res"] * 10**-6,
            "m",
        ],
        "y": [
            config["y_res"] * 10**-6,
            "m",
        ],
        "x": [
            config["x_res"] * 10**-6,
            "m",
        ],
        "t": [0.001, "s"],
    }

    projectionOrientation = [
        0.459884375333786,
        0.6998259425163269,
        -0.031935740262269974,
        0.5456465482711792,
    ]

    colors = []
    for channel_str in s3_channel_paths:
        channel_str = str(Path(channel_str).stem).replace(".ome", "")
        channel: int = int(channel_str.split("_")[-1])
        hex_val: int = wavelength_to_hex_alternate(channel)
        hex_code = f"#{str(hex(hex_val))[2:]}"
        hex_str = (
            '#uicontrol vec3 color color(default="'
            + hex_code
            + '")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}'
        )

        colors.append(hex_str)

    # Creating layer per channel
    layers = []
    for idx in range(len(s3_channel_paths)):
        channel_name = Path(s3_channel_paths[idx]).name

        layers.append(
            {
                "source": s3_channel_paths[idx],
                "type": "image",
                # use channel idx when source is the same
                # in zarr to change channel otherwise 0
                "channel": 0,
                "name": channel_name,
                "opacity": 1,
                "blend": "additive",
                "tab": "rendering",
                "shader": colors[idx],
                "shaderControls": {
                    "normalized": {
                        "range": [0, dynamic_ranges[channel_name][0]],
                        "window": [0, dynamic_ranges[channel_name][1]],
                    }
                },
            }
        )

    if segmentation:
        layers.append(
            {
                "source": f"precomputed://{s3_dataset_path}/image_atlas_alignment/ccf_annotation_precomputed",
                "type": "segmentation",
                "tab": "source",
                "name": "CCF_parcellation",
            }
        )

    if isinstance(orientation, dict):
        crossSectionOrientation = volume_orientation(orientation)
    else:
        crossSectionOrientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]

    subject_id = Path(s3_dataset_path).name.split("_")[1]
    crossSectionOrientation = volume_orientation(orientation)
    input_configs = {
        "title": subject_id,
        "dimensions": dimensions,
        "layers": layers,
        "crossSectionOrientation": crossSectionOrientation,
        "crossSectionScale": 15,
        "projectionScale": 10240,
        "projectionOrientation": projectionOrientation,
        "toolPalettes": {
            "Shader controls": {"row": 2, "query": "type:shaderControl"},
        },
    }

    json_state = utils.generate_ng_link(
        input_configs=input_configs,
        s3_path=s3_dataset_path,
        base_url=config["ng_base_url"],
        json_name="neuroglancer_config.json",
        segmentation=segmentation,
    )

    ng_output_path = f"{config['output_folder']}/neuroglancer_config.json"

    with open(ng_output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)

    return Path(ng_output_path), json_state["ng_link"]


def run():
    """
    Run function allows the smartspim pipeline to execute
    in parallel. It receives an input parameter related to
    the capsule mode:

    - "dispatch": This mode dispatches multiple instances of
    the downstream capsules.

    - "clean": This mode cleans up all the results from the
    downstream capsules because our data is being copied to the
    aind-open-data bucket.

    There are two more parameters useful to process data.

    - cloud_mode: Provide 'true' if you want to output data
    in the cloud, 'false' otherwise. If 'true', we only support
    AWS buckets and only the bucket and suffix must be provided.

    - output_path: Path where you want to output the processed
    dataset. If cloud_mode is 'true' then the data will be moved
    to a provided AWS bucket, 'false' means you will store the
    dataset locally. Be aware we are currently using cp command.
    """

    # Absolute paths of common Code Ocean folders
    data_folder = Path(os.path.abspath("../data"))
    results_folder = Path(os.path.abspath("../results"))

    params = str(sys.argv[1:])
    params = params.replace("[", "").replace("]", "").casefold()

    try:
        mode, cloud_mode, output_path = params.split(",")

    except ValueError as e:
        print(f"Three parameters are required as input!, error {e}")
        exit(1)

    cloud_mode = str_to_bool(cloud_mode)
    output_path = output_path.strip().replace("'", "")
    sys.argv = [sys.argv[0]]

    # It is assumed that these files
    # will be in the data folder
    required_input_elements = [
        f"{data_folder}/processing_manifest.json",
        f"{data_folder}/input_aind_metadata/data_description.json",
    ]

    if "clean" in mode:
        required_input_elements = [
            f"{data_folder}/modified_processing_manifest.json",
            f"{data_folder}/input_aind_metadata/data_description.json",
        ]

    if "postprocess-start" in mode:
        required_input_elements = [
            f"{data_folder}/raw_data",
            f"{data_folder}/stitched_data",
        ]

    if "postprocess-stop" in mode:
        required_input_elements = [
            f"{data_folder}/registration",
            f"{data_folder}/classification",
            f"{data_folder}/quantification",
            f"{data_folder}/postprocess_dispatch",
            f"{data_folder}/stitched_data",
        ]

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    logger.info(f"Data in data folder: {os.listdir(data_folder)}")
    logger.info(f"Mode: {mode} - Cloud mode: {cloud_mode} - Output path: {output_path}")

    if "dispatch" in mode:
        pipeline_config, dataset_name, investigators = get_data_config(
            data_folder=data_folder,
            data_description_path="input_aind_metadata/data_description.json",
        )

        # Creating new metadata for stitched dataset
        output_dispatch_metadata, new_dataset_name = create_derived_stitched_metadata(
            data_folder=data_folder, results_folder=results_folder, logger=logger
        )

        # Looking for files
        flatfield_folder = data_folder.joinpath("flatfield_estimation")
        destripe_files = [str(p) for p in list(data_folder.glob("image_destriping_*"))]
        stitch_folder = data_folder.joinpath("stitched")
        fuse_folder = data_folder.joinpath("fused")
        ccf_folders = glob(f"{data_folder}/ccf_registration_results/ccf_*")

        s3_path, dest_zarr_path = copy_intermediate_data(
            output_dispatch_metadata=output_dispatch_metadata,
            flatfield_folder=flatfield_folder,
            destripe_files=destripe_files,
            stitch_folder=stitch_folder,
            fuse_folder=fuse_folder,
            ccf_folders=ccf_folders,
            new_dataset_name=new_dataset_name,
            output_path=output_path,
            results_folder=results_folder,
            logger=logger,
        )

        # Getting S3 paths for channels
        s3_paths_for_channels = []
        for fuse_folder in fuse_folder.glob("*.zarr"):
            channel_name = f"{Path(fuse_folder).name}".replace("fusion_", "")
            # f"{s3_path}/{output_fusion}/OMEZarr"
            s3_paths_for_channels.append(f"{dest_zarr_path}/{channel_name}.zarr")

        chanel_dynamic_ranges = utils.calculate_dynamic_range(
            fuse_folder=fuse_folder, extension="*.zarr", percentile=99, level=3
        )
        orientation = pipeline_config["prelim_acquisition"]

        axes_resolution = pipeline_config["pipeline_processing"]["stitching"][
            "resolution"
        ]

        output_json, ng_link_path = create_neuroglancer_link(
            config={
                "bucket_path": output_path,
                "output_folder": results_folder,
                "ng_base_url": "https://neuroglancer-demo.appspot.com/",
                "z_res": axes_resolution[2]["resolution"],
                "y_res": axes_resolution[1]["resolution"],
                "x_res": axes_resolution[0]["resolution"],
            },
            s3_channel_paths=s3_paths_for_channels,
            s3_dataset_path=s3_path,
            orientation=orientation,
            dynamic_ranges=chanel_dynamic_ranges,
            segmentation=False,
        )

        # Creating QC Metrics
        qc_evaluators = [
            {
                "name": "Neuroglancer Link Evaluation",
                "description": "Checks that the whole-brain neuroglancer link was created",
                "notes": "",
                "stage": "Processing",
                "qc_metric_values": [
                    {
                        "name": "Dataset neuroglancer link",
                        "description": "Qualitative check that the neuroglancer link was created",
                        "value": "",
                        "reference": ng_link_path,
                        "status": "Pending",
                    },
                ],
            },
        ]

        utils.create_quality_control_metadata(
            qc_eval_values=qc_evaluators,
            output_path=output_dispatch_metadata,
        )

        data_results = glob(f"{results_folder}/*")
        logger.info(f"Data in {results_folder}: {data_results}")

        # Copying full res neuroglancer config out
        if cloud_mode:
            for out in utils.execute_command_helper(
                f"aws s3 cp {output_json} {s3_path}/{output_json.name}"
            ):
                logger.info(out)

        else:
            for out in utils.execute_command_helper(
                f"cp {output_json} {s3_path}/{output_json.name}"
            ):
                logger.info(out)

        output_json, ng_link_path = create_neuroglancer_link(
            config={
                "bucket_path": output_path,
                "output_folder": results_folder,
                "ng_base_url": "https://neuroglancer-demo.appspot.com/",
                "z_res": axes_resolution[2]["resolution"],
                "y_res": axes_resolution[1]["resolution"],
                "x_res": axes_resolution[0]["resolution"],
            },
            s3_channel_paths=s3_paths_for_channels,
            s3_dataset_path=s3_path,
            orientation=orientation,
            dynamic_ranges=chanel_dynamic_ranges,
            segmentation=True,
        )

        # Copying registration neuroglancer config out
        if cloud_mode:
            for out in utils.execute_command_helper(
                f"aws s3 cp {output_json} {s3_path}/image_atlas_alignment/{output_json.name}"
            ):
                logger.info(out)

        else:
            for out in utils.execute_command_helper(
                f"cp {output_json} {s3_path}/image_atlas_alignment/{output_json.name}"
            ):
                logger.info(out)

        # Setting the stitching path in pipeline config
        pipeline_config["pipeline_processing"]["stitching"]["s3_path"] = s3_path

        dispatch(
            processing_manifest=pipeline_config,
            results_folder=results_folder,
        )

        utils.save_dict_as_json(
            f"{results_folder}/modified_processing_manifest.json",
            pipeline_config,
        )

    elif "clean" in mode:
        logger.info("Starting cleaning...")
        pipeline_config, dataset_name, investigators = get_data_config(
            data_folder=data_folder,
            data_description_path="input_aind_metadata/data_description.json",
            processing_manifest_path="modified_processing_manifest.json",
        )

        pipeline_config["name"] = dataset_name

        clean_up(
            processing_manifest=pipeline_config,
            data_folder=data_folder,
            results_folder=results_folder,
            cloud_mode=cloud_mode,
        )

    else:
        raise NotImplementedError(f"The mode {mode} has not been implemented")


if __name__ == "__main__":
    run()
