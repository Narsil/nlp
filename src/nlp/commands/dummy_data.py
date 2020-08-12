import logging
import tempfile
import os
from argparse import ArgumentParser
import builtins
import zipfile
import shutil

from nlp.commands import BaseTransformersCLICommand
from nlp.load import import_main_class, prepare_module
from nlp.utils import MockDownloadManager
from nlp.utils import DownloadManager
from nlp.utils.file_utils import DownloadConfig


logger = logging.getLogger(__name__)


def slurp(filename, n_first, n_last):
    total_lines = 0
    with open(filename, "r") as f:
        for line in f:
            total_lines += 1

    lines = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            if i <= n_first or i >= total_lines - n_last:
                lines.append(line)
    return lines


def test_command_factory(args):
    return DummyDataCommand(args.path_to_dataset, args.requires_manual, args.auto_zip, args.n_first, args.n_last,)


class DummyDataCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        test_parser = parser.add_parser("dummy_data")
        test_parser.add_argument(
            "--requires_manual", action="store_true", help="Dataset requires manual data",
        )
        test_parser.add_argument(
            "--auto-zip",
            action="store_true",
            help="Attempts to automatically create the dummy files. Use with `--n-first`, `--n-last` to trim the real dataset files",
        )
        test_parser.add_argument(
            "--n-first",
            type=int,
            default=0,
            help="Automatically creates the test datasets by keeping the first n lines from the real datasets (requires `--auto-zip`)",
        ),
        test_parser.add_argument(
            "--n-last",
            type=int,
            default=0,
            help="Automatically creates the test datasets by keeping the last n lines from the real datasets (requires `--auto-zip`)",
        ),
        test_parser.add_argument("path_to_dataset", type=str, help="Name of the dataset to download")
        test_parser.set_defaults(func=test_command_factory)

    def __init__(
        self, path_to_dataset: str, requires_manual: bool, auto_zip: bool, n_first: int, n_last: int,
    ):
        self._path_to_dataset = path_to_dataset
        self._requires_manual = requires_manual
        self._dataset_name = path_to_dataset.split("/")[-2]
        if auto_zip:
            assert n_first or n_last, "When using `auto-zip` you need to set at least `--n-first` or `--n-last`"
        else:
            assert n_first == 0 and n_last == 0, "When using `--n-first` or `--n-last` you need to set `--auto-zip`"

        self._auto_zip = auto_zip
        self._n_first = n_first
        self._n_last = n_last

    def run(self):
        module_path, hash = prepare_module(self._path_to_dataset)
        builder_cls = import_main_class(module_path)

        # use `None` as config if no configs
        configs = builder_cls.BUILDER_CONFIGS or [None]

        for config in configs:
            if config is None:
                name = None
                version = builder_cls.VERSION
            else:
                version = config.version
                name = config.name

            dataset_builder = builder_cls(name=name, hash=hash)
            dummy_data_folder, files_to_create = self.get_expected_files(
                config, version, dataset_builder, print_dummy_instructions=not self._auto_zip,
            )
            if self._auto_zip:
                self.auto_zip(config, version, dataset_builder, dummy_data_folder, files_to_create)

    def get_real_files(self, dataset_builder):
        dl_manager = DownloadManager(dataset_name=self._dataset_name, download_config=DownloadConfig())
        return self.get_files(dataset_builder, dl_manager)

    def get_files(self, dataset_builder, dl_manager, mock=False):
        files = []

        def monkey_patch_open(*args, **kwargs):
            files.append(args[0])
            if mock:
                args = tuple(list(args)[1:])
                return tempfile.TemporaryFile(*args, **kwargs)
            else:
                return builtins._open(*args, **kwargs)

        builtins._open = builtins.open
        builtins.open = monkey_patch_open

        generator_splits = dataset_builder._split_generators(dl_manager)

        for split in generator_splits:
            gen_kwargs = split.gen_kwargs
            generator = dataset_builder._generate_examples(**gen_kwargs)
            for key, record in generator:
                pass

        builtins.open = builtins._open
        return files

    def auto_zip(self, config, version, dataset_builder, dummy_data_folder, files_to_create):

        real_files = self.get_real_files(dataset_builder)

        for file_to_create in files_to_create:
            try:
                real_file = [
                    filename
                    for filename in real_files
                    if os.path.basename(filename) == os.path.basename(file_to_create)
                ][0]
            except IndexError:
                raise Exception(f"We could not find the file {file_to_create} in the real downladed files")

            lines = slurp(real_file, self._n_first, self._n_last)

            full_filename_to_create = os.path.join(dummy_data_folder, file_to_create)
            os.makedirs(os.path.dirname(full_filename_to_create), exist_ok=True)

            with open(full_filename_to_create, "w") as f:
                for line in lines:
                    f.write(line)

        archive_name = "dummy_data.zip"
        # Create archive
        dummy_filename = os.path.join(dummy_data_folder, archive_name)
        logger.info(f"Creating dummy data archive {dummy_filename}")
        with zipfile.ZipFile(dummy_filename, "w") as z:
            for file_to_create in files_to_create:
                filename = os.path.join(dummy_data_folder, file_to_create)
                logger.info(f"Adding dummy file archive {filename}")
                z.write(filename, arcname=file_to_create)
        logger.info(f"Created archive")

        # Remove other files
        for filename in os.listdir(dummy_data_folder):
            if filename == archive_name:
                continue
            else:
                full_filename = os.path.join(dummy_data_folder, filename)
                if os.path.isdir(full_filename):
                    shutil.rmtree(full_filename)
                else:
                    os.remove(full_filename)

    def get_expected_files(self, config, version, dataset_builder, print_dummy_instructions=False):
        mock_dl_manager = MockDownloadManager(
            dataset_name=self._dataset_name, config=config, version=version, is_local=True,
        )
        dummy_data_folder = os.path.join(self._path_to_dataset, mock_dl_manager.dummy_data_folder)
        logger.info(f"Creating dummy folder structure for {dummy_data_folder}... ")
        os.makedirs(dummy_data_folder, exist_ok=True)

        try:
            generator_splits = dataset_builder._split_generators(mock_dl_manager)
        except FileNotFoundError as e:

            print(
                f"Dataset {self._dataset_name} with config {config} seems to already open files in the method `_split_generators(...)`. You might consider to instead only open files in the method `_generate_examples(...)` instead. If this is not possible the dummy data has to be created with less guidance. Make sure you create the file {e.filename}."
            )

        split_names = []
        dummy_file_name = mock_dl_manager.dummy_file_name

        for split in generator_splits:
            logger.info(f"Collecting dummy data file paths to create for {split.name}")
            split_names.append(split.name)

            dummy_data_guidance_print = "\n" + 30 * "=" + "DUMMY DATA INSTRUCTIONS" + 30 * "=" + "\n"
            config_string = f"config {config.name} of " if config is not None else ""
            dummy_data_guidance_print += (
                "- In order to create the dummy data for "
                + config_string
                + f"{self._dataset_name}, please go into the folder '{dummy_data_folder}' with `cd {dummy_data_folder}` . \n\n"
            )

            dummy_data_guidance_print += f"- It appears that the function `_generate_examples(...)` expects one or more files in the folder {dummy_file_name} using the function `glob.glob(...)`. In this case, please refer to the `_generate_examples(...)` method to see under which filename the dummy data files should be created. \n\n"

        files_to_create = self.get_files(dataset_builder, mock_dl_manager, mock=True)

        split_names = ", ".join(split_names)
        if print_dummy_instructions:
            if len(files_to_create) > 0:
                # no glob.glob(...) in `_generate_examples(...)`
                if len(files_to_create) == 1 and next(iter(files_to_create)) == dummy_file_name:
                    dummy_data_guidance_print += f"- Please create a single dummy data file called '{next(iter(files_to_create))}' from the folder '{dummy_data_folder}'. Make sure that the dummy data file provides at least one example for the split(s) '{split_names}' \n\n"
                    files_string = dummy_file_name
                else:
                    files_string = ", ".join(files_to_create)
                    dummy_data_guidance_print += f"- Please create the following dummy data files '{files_string}' from the folder '{dummy_data_folder}'\n\n"

                    dummy_data_guidance_print += f"- For each of the splits '{split_names}', make sure that one or more of the dummy data files provide at least one example \n\n"

                dummy_data_guidance_print += f"- If the method `_generate_examples(...)` includes multiple `open()` statements, you might have to create other files in addition to '{files_string}'. In this case please refer to the `_generate_examples(...)` method \n\n"

            if len(files_to_create) == 1 and next(iter(files_to_create)) == dummy_file_name:
                dummy_data_guidance_print += f"-After the dummy data file is created, it should be zipped to '{dummy_file_name}.zip' with the command `zip {dummy_file_name}.zip {dummy_file_name}` \n\n"

                dummy_data_guidance_print += (
                    f"-You can now delete the file '{dummy_file_name}' with the command `rm {dummy_file_name}` \n\n"
                )

                dummy_data_guidance_print += f"- To get the file '{dummy_file_name}' back for further changes to the dummy data, simply unzip {dummy_file_name}.zip with the command `unzip {dummy_file_name}.zip` \n\n"
            else:
                dummy_data_guidance_print += f"-After all dummy data files are created, they should be zipped recursively to '{dummy_file_name}.zip' with the command `zip -r {dummy_file_name}.zip {dummy_file_name}/` \n\n"

                dummy_data_guidance_print += f"-You can now delete the folder '{dummy_file_name}' with the command `rm -r {dummy_file_name}` \n\n"

                dummy_data_guidance_print += f"- To get the folder '{dummy_file_name}' back for further changes to the dummy data, simply unzip {dummy_file_name}.zip with the command `unzip {dummy_file_name}.zip` \n\n"

            dummy_data_guidance_print += (
                f"- Make sure you have created the file '{dummy_file_name}.zip' in '{dummy_data_folder}' \n"
            )

            dummy_data_guidance_print += 83 * "=" + "\n"

            print(dummy_data_guidance_print)
        return (dummy_data_folder, files_to_create)
