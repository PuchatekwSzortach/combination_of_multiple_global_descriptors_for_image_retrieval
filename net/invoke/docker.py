"""
Module with docker related commands
"""

import invoke


@invoke.task
def run(context, config_path):
    """
    Run docker container for the app

    :param context: invoke.Context instance
    :param config_path: str, path to configuration file
    """

    import os

    import net.utilities

    is_cuda_available = "/cuda/" in os.environ["PATH"]

    config = net.utilities.read_yaml(config_path)

    os.makedirs(os.path.dirname(config["log_path"]), exist_ok=True)

    # Don't like this line, but necessary to let container write to volume shared with host and host
    # to be able to read that data
    context.run(f'sudo chmod -R 777 {os.path.dirname(config["log_path"])}', echo=True)

    # Also need to give container access to .git repository
    context.run('sudo chmod -R 777 .git', echo=True)

    run_options = {
        "gpu_capabilities": "--gpus all" if is_cuda_available else ""
    }
    
    command = (
        "docker run -it --rm "
        "{gpu_capabilities} "
        "-v $PWD:/app:delegated "
        "-v $PWD/../../data:/data:delegated "
        "-v /tmp/logs:/tmp/logs:delegated "
        "puchatek_w_szortach/combination_of_multiple_global_descriptors:latest /bin/bash"
    ).format(**run_options)

    context.run(command, pty=True, echo=True)


@invoke.task
def build_app_container(context):
    """
    Build app container

    :param context: invoke.Context instance
    """

    command = (
        "docker build "
        "--tag puchatek_w_szortach/combination_of_multiple_global_descriptors:latest "
        "-f ./docker/app.Dockerfile ."
    )

    context.run(command, echo=True)


@invoke.task
def build_app_base_container(context, tag):
    """
    Build app base container

    :param context: invoke.Context instance
    :param context: tag: str, tag for the image
    """

    command = (
        "docker build "
        f"--tag puchatek_w_szortach/combination_of_multiple_global_descriptors_base:{tag} "
        "-f ./docker/app_base.Dockerfile ."
    )

    context.run(command, echo=True)
