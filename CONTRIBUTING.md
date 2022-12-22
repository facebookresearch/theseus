# Contributing to Theseus

We want to make contributing to Theseus as easy and transparent as possible.

### Developer Guide

- Fork the repo and install with development requirements.
  ```bash
  git clone <link_to_fork> && cd theseus
  pip install -e ".[dev]"
  ```
- Make a branch from `main`. See the [workflow model](#workflow-model) we follow.
  ```bash
  git checkout -b <username>.<appropriate_branch_name> main
  ````
- Make small, independent, and well documented commits. If you've changed the API, update the documentation. Add or modify unit tests as appropriate. Follow this [code style](#code-style).
- See [pull requests](#pull-requests) guide when you are ready to have your code reviewed to be merged into `main`. It will be included in the next release following this [versioning](#versioning).
- See [issues](#issues) for questions, suggestions, and bugs.
- If you haven't already, complete the [Contributor License Agreement](#contributor-license-agreement) and see [license](#license).

---

## Workflow Model

We follow the [Trunk-based](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development) model. Small and frequent PR of new features will be merged to `main` and a tagged release will indicate latest stable version on `main` history.

## Code Style

For Python we use `black` and `isort` for linting and code style, and use [typing](https://docs.python.org/3/library/typing.html). We also use pre-commit hooks to ensure linting and style enforcement.
```bash
pip install pre-commit && pre-commit install && pre-commit run --all-files
```

## Pull Requests

- We encourage more smaller and focused PRs rather than big PRs with many independent changes.
- Use this [PR template](.github/PULL_REQUEST_TEMPLATE.md) to submit your code for review. Consider using the [draft PR](https://github.blog/2019-02-14-introducing-draft-pull-requests/) option to gather early feedback.
- Add yourself to the `Assignees`, add [Mustafa Mukadam](https://github.com/mhmukadam) and [Luis Pineda](https://github.com/luisenp) as reviewers, link to any open issues that can be closed when the PR is merged, and add appropriate `Labels` and `Milestone`.
- We expect the PR is ready for final review only if Continuous Integration tests are passing.
- Keep your branch up-to-date with `main` by rebasing as necessary.
- We employ `squash-and-merge` for incorporating PRs. Add a brief change summary to the commit message.
- After PR is approved and merged into `main`, delete the branch to reduce clutter.

## Versioning

We use [semantic versioning](https://semver.org/). For core Theseus team member, to prepare a release:
- Update version in [init](https://github.com/facebookresearch/theseus/blob/main/theseus/__init__.py) file.
- Make sure all tests are passing.
- Create a release tag with changelog summary using the github release interface.

## Issues

We use [GitHub issues](https://github.com/facebookresearch/theseus/issues) to track bugs. You can also reach out to us there with questions or suggestions. Please ensure your description is clear by following our [templates](https://github.com/facebookresearch/theseus/issues/new/choose).

## Contributor License Agreement

In order to accept your pull request, we need you to submit a Contributor License Agreement (CLA). You only need to do this once to work on any of Meta's open source projects. Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to Theseus, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
