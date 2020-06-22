"""
Module with docker related commands
"""

import invoke


@invoke.task
def run(context):
    """
    Run docker container for the app

    :param context: invoke.Context instance
    """

    command = (
        "docker run -it --rm "
        "-v $PWD:/app:delegated "
        "-v $PWD/../../data:/data:delegated "
        "-v /tmp/logs:/tmp/logs:delegated "
        "puchatek_w_szortach/combination_of_multiple_global_descriptors:latest bash"
    )

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
