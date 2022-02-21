"""defaults."""
import argparse
import os

from fvcore.common.file_io import PathManager

from struct_discovery.utils.logger import setup_logger
from struct_discovery.utils.env import seed_all_rng


def default_argument_parser():
    parser = argparse.ArgumentParser(description="Structure Discovery")
    parser.add_argument("--config-file", default="",
                        metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to attempt to resume from the checkpoint directory",
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="perform evaluation only")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def default_setup(cfg, args):
    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        PathManager.mkdirs(output_dir)

    setup_logger(output_dir, name="fvcore")
    logger = setup_logger(output_dir)

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(
                    args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))

    if output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(path))

    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED)
