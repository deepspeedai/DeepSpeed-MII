# Contributing
DeepSpeed-MII welcomes your contributions!

## Prerequisites
We use [pre-commit](https://pre-commit.com/) to ensure that formatting is
consistent across DeepSpeed. First, ensure that `pre-commit` is installed from either
installing DeepSpeed or `pip install pre-commit`. Next, the pre-commit hooks must be
installed once before commits can be made:
```bash
pre-commit install
```

Afterwards, our suite of formatting tests run automatically before each `git commit`. You
can also run these manually:
```bash
pre-commit run --all-files
```
If a formatting test fails, it will fix the modified code in place and abort
the `git commit`. After looking over the changes, you can `git add <modified files>`
and then repeat the previous `git commit` command.

## Developer Certificate of Origin
This project welcomes contributions and suggestions. All contributions to deepspeedai projects
require commits to be signed off with a [Developer Certificate of Origin](https://en.wikipedia.org/wiki/Developer_Certificate_of_Origin)
(DCO) declaring that you have the right to, and actually do, grant us the rights to use your contribution.

When you submit a pull request, the DCO app will check for the presence of signed commits.
Information about how this check works is here: https://github.com/dcoapp/app?tab=readme-ov-file#how-it-works

## Code of Conduct
This project has adopted the [Microsoft Open Source Code of
Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact
[opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or
comments.
