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
    context.run("xenon . --max-absolute B", echo=True)
    context.run("pylint {}".format(directories), echo=True)


@invoke.task
def commit_stage(context):
    """
    Run commit stage tasks

    :param context: invoke.Context instance
    """

    unit_tests(context)
    static_code_analysis(context)
    inserts_count_check(context)


@invoke.task
def inserts_count_check(context):
    """
    Check current tree doesn't have too many changes w.r.t. origin/master

    :param context: invoke.Context instance
    """

    import git
    import pydriller

    def get_target_commits_hashes():
        """
        Get commit hashes of origin/master and head

        :return: tuple of two strings
        """

        repository = git.Repo(".")
        repository.remote().fetch()

        master = repository.commit("origin/master")
        head = repository.commit("head")

        return master.hexsha, head.hexsha

    def should_modification_be_ignored(modification):
        """
        Simple helper for filtering out git modifications that shouldn't be counted towards insertions check.
        Filters out tools configuration files and similar.

        :param modification: pydriller.domain.commit.Modification instance
        :return: bool
        """

        patterns = [
            ".devcontainer",
            ".pylintrc",
            ".gitignore"
        ]

        for pattern in patterns:

            if pattern in modification.new_path:

                return True

        return False

    master_sha, head_sha = get_target_commits_hashes()

    repository_mining = pydriller.RepositoryMining(
        path_to_repo=".",
        from_commit=master_sha,
        to_commit=head_sha
    )

    additions_count = 0

    for commit in repository_mining.traverse_commits():

        for modification in commit.modifications:

            if should_modification_be_ignored(modification) is False:

                additions_count += modification.added

    threshold = 300

    print(f"Inserts between origin/master and HEAD: {additions_count}/{threshold}")

    if additions_count > threshold:

        raise ValueError("Exceeded max inserts count")
