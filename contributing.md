### Thanks for Your Help!

Contributing is so kind of you. In `annsa`, all contributions, bug reports, bug
fixes, documentation improvements, enhancements and ideas are welcome.

The [GitHub "issues" tab](https://github.com/annsa/annsa/issues)
contains some issues labeled as beginner issues. Those are open issues that
would be a good quick way to get started. Browse them to see if you want to get
started on one.  

#### Bug Reports

To report a bug, please create a [new GitHub 
issue](https://github.com/annsa/annsa/issues/new)

  - Please include a short but detailed, self-contained Python snippet or
    explanation for reproducing the problem.

  - Explain what the expected behavior was, and what you observed instead.

  - To get our attention, please @-message one of the developers.

##### Instructions for setting up a development environment

The `annsa` project aims to to be compatible with Python 3.  Anaconda is the 
recommended distribution to use to work on annsa; we will
assume that if you want to use another distribution or your own set up, you can
translate the instructions. It is important to install `annsa` and to run the
tests on your platform before beginning to work.

You can download Anaconda at https://www.continuum.io/Downloads for the full
install. You can also download a mini Anaconda install for a bare-bones
install -- this is good for a build server or if you don't have much space.
The mini Anaconda installs are available at https://conda.io/miniconda.html.

Once your Anaconda package is installed and available, we recommend that you 
create a conda environment just for `annsa` development. Start with:

```bash
conda create -q -n annsa-36-test-env python=3.6 requirements.txt
```

This command will install the packages in the requirements.txt file and may
take a bit of time -- give it a few minutes to download and install the
packages and their dependences. Once complete, switch to your new environment
and install additional packages needed to run and test.

Activate the 3.6 environment and install `annsa`:
 
```
source activate annsa-36-test-env
python setup.py install
```

##### Run the tests

Tests are automatically detected and run with pytest. To run them, use
the pytest tool that was made available when pytest was installed
and add the `annsa` code into the PYTHONPATH variable so that the tests
can find the implementation code.

Start in the root directory where you have cloned the annsa repository
and run for Python 3.6 --

```bash
source active annsa-36-test-env
pytest annsa
```

##### Run style tests with flake8

Adherance to style checks in flake8 is encouraged though not strictly
enforced. While you should not feel compelled to fix existing failures,
please do not add additional flake8 issues.

  - run flake8 from the root of the annsa working directory to get all flake8 
    issues
  - run flake8 and provide a filename to just run checks on that file
  
##### Pull Requests

  - **Make sure the test suite passes** on your computer. To do so, run 
    `py.test` in the tests directory.
  - Please reference relevant Github issues in your commit message using 
    `GH1234` or `#1234`.
  - Changes should be PEP8 compatible 
    [PEP8](http://www.python.org/dev/peps/pep-0008/).
  - If a PR has multiple separable concerns, it is too comples. Please submit 
    separate PRs accordingly.
  - Keep style fixes to a separate commit to make your PR more readable.
  - Docstrings ideally follow 
    [sphinx autodoc](https://pythonhosted.org/an_example_pypi_project/sphinx.html#function-definitions)
  - Write tests. For help, consult
    [this tutorial on how to write tests](http://katyhuff.github.io/python-testing/).
  - When writing tests, please make sure they are in a `test` directory.
  - When you start working on a PR, start by creating a new branch pointing at 
    the latest commit on github master.
  - Please avoid rebasing if possible. Nothing wrong with rebase... it is just 
    confusing for @katyhuff .
  - The `annsa` copyright policy is detailed in the `annsa` 
    [LICENSE](https://github.com/annsa/annsa/blob/master/LICENSE).

#### More developer docs

* We are working on it.


#### Meta
Note, this contributing file was adapted from the one at the
[pandas](https://github.com/pydata/pandas) repo. Thanks pandas!

