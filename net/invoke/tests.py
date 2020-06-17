"""Module with tests tasks"""

import invoke


@invoke.task
def unit_tests(context):
    """Run unit tests

    :param context: invoke.Context instance
    """

    context.run("pytest ./tests", pty=True, echo=True)


@invoke.task
def static_code_analysis(context):
    """Run static code analysis

    :param context: invoke.Context instance
    """

    directories = "net tests"

    context.run("pycodestyle {}".format(directories), echo=True)
    context.run("pylint {}".format(directories), echo=True)
    context.run("xenon . --max-absolute B", echo=True)


@invoke.task
def commit_stage(context):
    """Run commit stage tasks

    :param context: invoke.Context instance
    """

    unit_tests(context)
    static_code_analysis(context)
