# Contributing to Theseus

We want to make contributing to Theseus as easy and transparent as possible.

### Developer Guide

- Fork the repo and make a branch from `develop`. See the hybrid [workflow model](#workflow-model) we follow.
  ```bash
  git checkout -b <username>.<appropriate_branch_name> develop
  ````
- Make small, independent, and well documented commits. If you've changed APIs, update the documentation. Add or modify unit tests as appropriate. Follow this [code style](#code-style).
- See [pull requests](#pull-requests) guide when you are ready to have your code reviewed to be merged into `develop`. It will be included in the next release on `main` following this [versioning](#versioning).
- See [issues](#issues) for questions, suggestions, and bugs.
- If you haven't already, complete the [Contributor License Agreement](#contributor-license-agreement) and see [license](#license).

---

## Workflow Model

We follow a hyrbid between [Gitflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow) and [Trunk-based](https://www.atlassian.com/continuous-delivery/continuous-integration/trunk-based-development) models. From the former we adopt hosting latest stable release on `main` branch and feature development on `develop` branch, and from the latter we adopt small and frequent merges of new features into `develop`.

## Code Style

For Python we use `black` and `isort` for linting and code style, and use [typing](https://docs.python.org/3/library/typing.html). We also use pre-commit hooks to ensure linting and style enforcement.
```bash
pip install pre-commit && pre-commit install && pre-commit run --all-files
```

## Pull Requests

- We encourage more smaller and focused PRs rather than big PRs with many independent changes.
- Use this [PR template](.github/PULL_REQUEST_TEMPLATE.md) to submit your code for review. Consider using the [draft PR](https://github.blog/2019-02-14-introducing-draft-pull-requests/) option to gather early feedback.
- Add yourself to the `Assignees`, add at least one core Theseus team member to `Reviewers`, link to any open issues that can be closed when the PR is merged, and add appropriate `Labels` and `Milestone`.
- We expect the PR is ready for final review only if Continuous Integration tests are passing.
- Keep your branch up-to-date with `develop` by rebasing as necessary.
- We employ `squash-and-merge` for incorporating PRs. Add a brief change summary to the commit message.
- After PR is approved and merged into `develop`, delete the branch to reduce clutter.

## Versioning

We use [semantic versioning](https://semver.org/). For core Theseus team member, to prepare a release:
- Update [version](version.txt) file.
- Make sure all tests are passing.
- Create a release tag with changelog summary using the github release interface.

## Issues

We use [GitHub issues](https://github.com/facebookresearch/theseus/issues) to track bugs. You can also reach out to us there with questions or suggestions. Please ensure your description is clear by following our [templates](https://github.com/facebookresearch/theseus/issues/new/choose).

## Contributor License Agreement

In order to accept your pull request, we need you to submit a Contributor License Agreement (CLA). You only need to do this once to work on any of Meta's open source projects. Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to Theseus, you agree that your contributions will be licensed under the [LICENSE](LICENSE) file in the root directory of this source tree.
