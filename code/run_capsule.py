""" Main script that works as a dispatcher in code ocean """

import json
import logging
import os
import re
import sys
from glob import glob
from pathlib import Path
from typing import List, Tuple, Union

import yaml
from ng_link import NgState

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

PIPELINE_VERSION = "2.0.2"
SCRIPT_DIR = Path(os.path.abspath(__file__)).parent


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


def get_yaml_config(filename):
    """
    Get default configuration from a YAML file.
    Parameters
    ------------------------
    filename: str
        String where the YAML file is located.
    Returns
    ------------------------
    Dict
        Dictionary with the configuration
    """

    with open(filename, "r") as stream:
        config = yaml.safe_load(stream)

    return config

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

        if not len(segment_channels):
            raise BaseException("Stopping pipeline, no segmentation channels.")

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
        raise BaseException("Stopping pipeline, pipeline configuration.")


def clean_up(
    processing_manifest: dict,
    data_folder: PathLike,
    results_folder: PathLike,
    cloud_mode: bool
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
    output_filename = utils.compile_processing_jsons(
        processing_paths=processing_paths,
        output_general_processing=results_folder,
        processor_full_name="Camilo Laiton",
        pipeline_version=PIPELINE_VERSION,
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
        pass

    utils.save_string_to_txt(
        f"Results of cell segmentation saved in: {cell_s3_output}",
        f"{results_folder}/output_cell.txt",
    )

    utils.save_string_to_txt(
        f"Results of quantification saved in: {quantification_s3_output}",
        f"{results_folder}/output_quantification.txt",
    )


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


def copy_intermediate_data(
    output_dispatch_metadata: PathLike,
    destripe_files: List[PathLike],
    flatfield_channels: List[PathLike],
    stitch_folders: List[PathLike],
    fuse_folders: List[PathLike],
    ccf_folders: List[PathLike],
    new_dataset_name: str,
    output_path: str,
    results_folder: PathLike,
    logger: logging.Logger,
    cloud_mode: bool,
) -> str:
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

    new_dataset_name: str
        New dataset name where the data will
        be copied following the aind conventions
        e.g., s3://{bucket_path}/{new_dataset_name}

    output_path: str
        Path where the data will be moved.
        Do not include 's3://' since this is
        automatically added if it's a S3 path

    results_folder: PathLike
        Results folder path in Code Ocean

    logger: logging.Logger
        Logging object

    cloud_mode: bool
        If the pipeline wants to output data
        in the cloud or locally. True for cloud,
        False for local.

    Returns
    -------
    Tuple[str, str]
        The first position is the path where the dataset
        was moved. e.g., s3://{bucket_path}/{new_dataset_name}
        It includes the "s3://" prefix. The second position
        is the folder inside that path where the Zarrs
        were moved.
        e.g., s3://{bucket_path}/{new_dataset_name}/{output_fusion}/OMEZarr
    """

    stitch_processings = []
    fuse_processings = []
    ccf_processings = []

    for stitch_folder in stitch_folders:
        processing_jsons = [
            p
            for p in glob(f"{stitch_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        stitch_processings.append(processing_jsons)

    for fuse_folder in fuse_folders:
        processing_jsons = [
            p
            for p in glob(f"{fuse_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        fuse_processings.append(processing_jsons)

    for ccf_folder in ccf_folders:
        processing_jsons = [
            p
            for p in glob(f"{ccf_folder}/metadata/*processing*.json")
            if "manifest" not in str(p)
        ]
        ccf_processings.append(processing_jsons)

    # Flattening list
    processing_paths = list()
    combined_processing_list = stitch_processings + fuse_processings + ccf_processings
    for sub_list in combined_processing_list:
        processing_paths += sub_list

    processing_paths = destripe_files + processing_paths
    logger.info(f"Processing paths: {processing_paths}")

    try:
        output_filename = utils.compile_processing_jsons(
            processing_paths=processing_paths,
            output_general_processing=output_dispatch_metadata,
            processor_full_name="Camilo Laiton",
            pipeline_version=PIPELINE_VERSION,
        )

    except Exception as e:
        print(f"Error while compiling processing manifests: {e}")
        output_filename = None

    logger.info(f"Compiled processing.json in path {output_filename}")

    if cloud_mode:
        s3_path = f"s3://{output_path}/{new_dataset_name}"

        # Copying derived metadata
        output_dispatch_metadata = Path(output_dispatch_metadata)
        for out in utils.execute_command_helper(
            f"aws s3 cp --recursive {output_dispatch_metadata} {s3_path}"
        ):
            logger.info(out)

        # Copying out fused data
        output_fusion = "image_tile_fusing"
        dest_zarr_path = f"{s3_path}/{output_fusion}/OMEZarr"
        dest_metadata_path = f"{s3_path}/{output_fusion}/metadata"

        for flatfield_channel in flatfield_channels:
            flatfield_channel_name = Path(flatfield_channel).name
            logger.info(
                f"Copying data from {flatfield_channel} to"
                f"{dest_metadata_path}/flatfield_correction/{flatfield_channel_name}"
            )
            for out in utils.execute_command_helper(
                f"aws s3 cp --recursive {flatfield_channel} {dest_metadata_path}/flatfield_correction/{flatfield_channel_name}"
            ):
                logger.info(out)

        for fuse_folder in fuse_folders:
            logger.info(f"Copying data from {fuse_folder} to {s3_path}/{output_fusion}")
            fuse_folder = Path(fuse_folder)
            source_zarr = fuse_folder.joinpath("OMEZarr")
            source_metadata = fuse_folder.joinpath("metadata")

            if source_zarr.exists():
                for out in utils.execute_command_helper(
                    f"aws s3 cp --recursive {source_zarr} {dest_zarr_path}"
                ):
                    logger.info(out)

            else:
                raise ValueError(f"Folder {source_zarr} does not exist!")

            if source_metadata.exists():
                for out in utils.execute_command_helper(
                    f"aws s3 cp --recursive {source_metadata} {dest_metadata_path}/{fuse_folder.name}"
                ):
                    logger.info(out)

            else:
                raise ValueError(f"Folder {source_metadata} does not exist!")

        # Copying stitch metadata
        for stitch_folder in stitch_folders:
            logger.info(f"Copying data from {stitch_folder} to {dest_metadata_path}")
            stitch_folder = Path(stitch_folder)
            source_metadata = stitch_folder.joinpath("metadata")

            if source_metadata.exists():
                for out in utils.execute_command_helper(
                    f"aws s3 cp --recursive {source_metadata} {dest_metadata_path}/{stitch_folder.name}"
                ):
                    logger.info(out)

            else:
                raise ValueError(f"Folder {source_metadata} does not exist!")

        # Copying ccf data
        ccf_s3_output = f"{s3_path}/image_atlas_alignment"
        regex_channels = r"Ex_(\d{3})_Em_(\d{3})$"

        for ccf_folder in ccf_folders:
            channel_name = re.search(regex_channels, ccf_folder).group()

            for out in utils.execute_command_helper(
                f"aws s3 mv --recursive {ccf_folder} {ccf_s3_output}/{channel_name}"
            ):
                logger.info(out)

        utils.save_string_to_txt(
            f"Stitched dataset saved in: {s3_path}",
            f"{results_folder}/output_stitching.txt",
        )
    
    else:
        # Organize files locally
        pass

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


def create_ng_link(
    config: dict, s3_channel_paths: List[str], s3_dataset_path: str
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
        "z": {
            "voxel_size": config["z_res"],
            "unit": "microns",
        },
        "y": {
            "voxel_size": config["y_res"],
            "unit": "microns",
        },
        "x": {
            "voxel_size": config["x_res"],
            "unit": "microns",
        },
        "t": {"voxel_size": 0.001, "unit": "seconds"},
    }

    colors = []
    for channel_str in s3_channel_paths:
        channel_str = Path(channel_str).stem
        channel: int = int(channel_str.split("_")[-1])
        hex_val: int = wavelength_to_hex_alternate(channel)
        hex_str = f"#{str(hex(hex_val))[2:]}"

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
                "shader": {
                    "color": colors[idx],
                    "emitter": "RGB",
                    "vec": "vec3",
                },
                "shaderControls": {"normalized": {"range": [0, 200]}},  # Optional
            }
        )

    subject_id = Path(s3_dataset_path).name.split("_")[1]
    input_configs = {
        "title": subject_id,
        "dimensions": dimensions,
        "layers": layers,
        "crossSectionOrientation": [0.5, 0.5, 0.5, -0.5],
        "crossSectionScale": 15,
    }

    neuroglancer_link = NgState(
        input_config=input_configs,
        mount_service="s3",
        bucket_path=config["bucket_path"],
        output_dir=config["output_folder"],
        base_url=config["ng_base_url"],
        json_name="neuroglancer_config.json",
    )

    ng_link = f"{config['ng_base_url']}#!{s3_dataset_path}/neuroglancer_config.json"
    # Modifying output path in s3 for when the data is moved
    json_state = neuroglancer_link.state
    json_state["ng_link"] = ng_link

    ng_output_path = f"{config['output_folder']}/neuroglancer_config.json"

    with open(ng_output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)

    return Path(ng_output_path), ng_link

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
        mode, cloud_mode, output_path = params.split(',')
    
    except ValueError as e:
        print(f"Three parameters are required as input!, error {e}")
        exit(1)

    cloud_mode = bool(cloud_mode)
    sys.argv = [sys.argv[0]]
    
    # Loading .env file
    # dotenv_path = Path(os.path.dirname(os.path.realpath(__file__))) / ".env"
    # load_env_file = load_dotenv(dotenv_path=dotenv_path)
    # logger.info(f"Load env file status: {load_env_file}")

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

    missing_files = utils.validate_capsule_inputs(required_input_elements)

    if len(missing_files):
        raise ValueError(
            f"We miss the following files in the capsule input: {missing_files}"
        )

    logger.info(f"Data in data folder: {os.listdir(data_folder)}")

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
        destripe_files = glob(f"{data_folder}/image_destriping_*")
        flatfield_channels = glob(f"{data_folder}/flatfield_correction_*")
        stitch_folders = glob(f"{data_folder}/stitched/stitch_*")
        fuse_folders = glob(f"{data_folder}/fused/fusion_*")
        ccf_folders = glob(f"{data_folder}/ccf_registration_results/ccf_*")

        s3_path, s3_dest_zarr = copy_intermediate_data(
            output_dispatch_metadata=output_dispatch_metadata,
            destripe_files=destripe_files,
            flatfield_channels=flatfield_channels,
            stitch_folders=stitch_folders,
            fuse_folders=fuse_folders,
            ccf_folders=ccf_folders,
            new_dataset_name=new_dataset_name,
            output_path=output_path,
            results_folder=results_folder,
            logger=logger,
            cloud_mode=cloud_mode,
        )

        # Getting S3 paths for channels
        s3_paths_for_channels = []
        for fuse_folder in fuse_folders:
            channel_name = f"{Path(fuse_folder).name}".replace("fusion_", "")
            # f"{s3_path}/{output_fusion}/OMEZarr"
            s3_paths_for_channels.append(f"{s3_dest_zarr}/{channel_name}.zarr")

        axes_resolution = pipeline_config["pipeline_processing"]["stitching"][
            "resolution"
        ]
        output_json, ng_link_path = create_ng_link(
            config={
                "bucket_path": output_path,
                "output_folder": results_folder,
                "ng_base_url": "https://aind-neuroglancer-sauujisjxq-uw.a.run.app",
                "z_res": axes_resolution[2]["resolution"],
                "y_res": axes_resolution[1]["resolution"],
                "x_res": axes_resolution[0]["resolution"],
            },
            s3_channel_paths=s3_paths_for_channels,
            s3_dataset_path=s3_path,
        )

        data_results = glob(f"{results_folder}/*")
        logger.info(f"Data in {results_folder}: {data_results}")

        if cloud_mode:
            # Copying neuroglancer config out
            for out in utils.execute_command_helper(
                f"aws s3 cp {output_json} {s3_path}/{output_json.name}"
            ):
                logger.info(out)
        
        else:
            # TODO: copy the neuroglancer config file
            pass

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
