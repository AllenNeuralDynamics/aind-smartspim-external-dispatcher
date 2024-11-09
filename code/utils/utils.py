"""
Utility functions
"""

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Union

from aind_data_schema.core.data_description import (DerivedDataDescription,
                                                    Funding)
from aind_data_schema.core.processing import (DataProcess, PipelineProcess,
                                              Processing)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.organizations import Organization
from aind_data_schema_models.pid_names import PIDName
from aind_data_schema_models.platforms import Platform
from pydantic import TypeAdapter

# IO types
PathLike = Union[str, Path]


def copy_file(input_filename: PathLike, output_filename: PathLike):
    """
    Copies a file to an output path

    Parameters
    ----------
    input_filename: PathLike
        Path where the file is located

    output_filename: PathLike
        Path where the file will be copied
    """

    try:
        shutil.copy(input_filename, output_filename)

    except shutil.SameFileError:
        raise shutil.SameFileError(
            f"The filename {input_filename} already exists in the output path."
        )

    except PermissionError:
        raise PermissionError(
            "Not able to copy the file. Please, check the permissions in the output path."
        )


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def delete_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Delete a folder path.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path that will be removed.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    shutil.Error:
        If the folder could not be removed.

    """
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            if verbose:
                print(f"Folder {dest_dir} was removed!")
        except shutil.Error as e:
            print(f"Folder could not be removed! Error {e}")


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
):
    """
    Execute a shell command.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(config: dict) -> None:
    """
    Execute a shell command with a given configuration.

    Parameters
    ------------------------

    command: str
        Command that we want to execute.

    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------

    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """
    # Command is not executed when info
    # is True
    if config["info"]:
        config["logger"].info(config["command"])
    else:
        for out in execute_command_helper(
            config["command"], config["verbose"], config["stdout_log_file"]
        ):
            if len(out):
                config["logger"].info(out)

            if config["exists_stdout"]:
                save_string_to_txt(out, config["stdout_log_file"], "a")


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------

    obj: object
        Object that wants to be validated.

    Returns
    ------------------------

    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------

    filename: str
        Name of the json file.

    dictionary: dict
        Dictionary that will be saved as json.

    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------

    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def check_type_helper(value: Any, val_type: type) -> bool:
    """
    Checks if a value belongs to a specific type.

    Parameters
    ------------------------

    value: Any
        variable data.

    val_type: type
        Type that we want to check.

    Returns
    ------------------------

    bool:
        True if the type is what we expect
        from the variable data, False otherwise.
    """

    if not isinstance(type(value), val_type):
        return False

    return True


def generate_timestamp(time_format: str = "%Y-%m-%d_%H-%M-%S") -> str:
    """
    Generates a timestamp in string format.

    Parameters
    ------------------------
    time_format: str
        String following the conventions
        to generate the timestamp (https://strftime.org/).

    Returns
    ------------------------
    str:
        String with the actual datetime
        moment in string format.
    """
    return datetime.now().strftime(time_format)


def validate_capsule_inputs(input_elements: List[str]) -> List[str]:
    """
    Validates input elemts for a capsule in
    Code Ocean.

    Parameters
    -----------
    input_elements: List[str]
        Input elements for the capsule. This
        could be sets of files or folders.

    Returns
    -----------
    List[str]
        List of missing files
    """

    missing_inputs = []
    for required_input_element in input_elements:
        required_input_element = Path(required_input_element)

        if not required_input_element.exists():
            missing_inputs.append(str(required_input_element))

    return missing_inputs

def generate_data_description(
    raw_data_description_path,
    dest_data_description,
    process_name: Optional[str] = "stitched",
):
    """
    Generates data description for the output folder.

    Parameters
    -------------

    raw_data_description_path: PathLike
        Path where the data description file is located.

    dest_data_description: PathLike
        Path where the new data description will be placed.

    process_name: str
        Process name of the new dataset


    Returns
    -------------
    str
        New folder name for the fused
        data
    """

    f = open(raw_data_description_path, "r")
    data = json.load(f)

    if isinstance(data["institution"], dict) and "abbreviation" in data["institution"]:
        institution = data["institution"]["abbreviation"]

    investigators = data["investigators"]

    if len(investigators) and len(investigators[0]):
        investigators = [PIDName.parse_obj(inv) for inv in investigators]

    else:
        investigators = [PIDName(name="Unknown")]

    # from_data_description
    funding_adapter = TypeAdapter(Funding)
    funding_sources = [
        funding_adapter.validate_python(fund) for fund in data["funding_source"]
    ]
    # Ensuring backwards compatibility
    derived = DerivedDataDescription(
        creation_time=datetime.now(),
        input_data_name=data["name"],
        process_name=process_name,
        institution=Organization.from_abbreviation(institution),
        funding_source=funding_sources,
        group=data["group"],
        investigators=investigators,
        platform=Platform.SMARTSPIM,
        project_name=data["project_name"],
        restrictions=data["restrictions"],
        modality=[Modality.SPIM],
        subject_id=data["subject_id"],
    )

    # derived.write_standard_file(output_directory=dest_data_description)
    with open(f"{dest_data_description}/data_description.json", "w") as f:
        f.write(derived.model_dump_json())

    return derived.name


def copy_available_metadata(
    input_path: PathLike, output_path: PathLike, files_to_copy: List[str]
) -> List[PathLike]:
    """
    Copies all the valid metadata from the aind-data-schema
    repository that exists in a given path.

    Parameters
    -----------
    input_path: PathLike
        Path where the metadata is located

    output_path: PathLike
        Path where we will copy the found
        metadata

    files_to_copy: List[str]
        List with the filenames of the metadata
        that we need to copy to the fused asset

    Returns
    --------
    List[PathLike]
        List with the metadata files that
        were copied
    """

    print("Files to copy: ", files_to_copy)
    # Making sure the paths are pathlib objects
    input_path = Path(input_path)
    output_path = Path(output_path)

    found_metadata = []

    for metadata_filename in files_to_copy:
        metadata_filename = input_path.joinpath(metadata_filename)

        if metadata_filename.exists():
            found_metadata.append(metadata_filename)

            # Copying file to output path
            output_filename = output_path.joinpath(metadata_filename.name)
            copy_file(metadata_filename, output_filename)

    return found_metadata


def generate_processing(
    data_processes: List[DataProcess],
    dest_processing: str,
    processor_full_name: str,
    pipeline_version: str,
) -> str:
    """
    Generates data description for the output folder.

    Parameters
    ------------------------

    data_processes: List[dict]
        List with the processes aplied in the pipeline.

    dest_processing: PathLike
        Path where the processing file will be placed.

    processor_full_name: str
        Person in charged of running the pipeline
        for this data asset

    pipeline_version: str
        Terastitcher pipeline version

    Returns
    -------
    str
        Path where the processing json was saved
    """
    # flake8: noqa: E501
    processing_pipeline = PipelineProcess(
        data_processes=data_processes,
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
        pipeline_url="https://github.com/AllenNeuralDynamics/aind-smartspim-pipeline",
    )

    processing = Processing(
        processing_pipeline=processing_pipeline,
        notes="This processing was generated by compiling independent processing jsons for every step",
    )

    processing.write_standard_file(output_directory=dest_processing)

    return dest_processing


def compile_processing_jsons(
    processing_paths: List[str],
    output_general_processing: str,
    processor_full_name: str,
    pipeline_version: str,
) -> str:
    """
    processing_paths: List[str]
        Paths where the processing jsons are located.
        The order of the list determines which data
        process goes first into the final processing.json

    output_general_processing: str
        Path where we will output the general processing
        json

    processor_full_name: str
        Person in charged to run the pipeline

    pipeline_version: str
        Pipeline version

    Returns
    -------
    str:
        Path where the processing json was saved
    """
    data_processes = []
    for processing_path in processing_paths:
        curr_processing = read_json_as_dict(str(processing_path))
        processing_adapter = TypeAdapter(Processing)
        curr_processing_obj = processing_adapter.validate_python(curr_processing)

        for data_process in curr_processing_obj.processing_pipeline.data_processes:
            data_processes.append(data_process)

    output_filename = generate_processing(
        data_processes=data_processes,
        dest_processing=str(output_general_processing),
        processor_full_name=processor_full_name,
        pipeline_version=pipeline_version,
    )

    return output_filename

def clean_investigator_names(investigators):
    """
    Formats investigator list to be syntactically correct

    Parameters
    ----------
    investigators : list
        list of investigator names

    Returns
    -------
    invest : str
        investigator names with proper syntax

    """

    if len(investigators) > 1:
        invest = ", ".join(investigators)
        idx = invest.rfind(",")
        invest = invest[:idx] + " and" + invest[idx + 1 :]
    else:
        invest = investigators[0]

    return invest
