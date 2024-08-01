# dstl-mast

To use MAST, change directory to the submodule:

```sh
$ cd dstl-mast
```

Then follow the installation instructions. You can then run code from dstl-mast.

## Install instructions  <a id="install-instructions"></a>


Using **pip** and **conda**:
```sh
$ conda create --name myenv python=3.9
$ conda activate myenv
$ pip install -r requirements.txt
$ cd StoneSoup
$ pip install -e .
```

## Using the RL plugin

The RL plugin provided by MAST can be accessed within the dstl-mast folder by import items from the stone soup plugins, for example:
```sh
$ from stonesoup.plugins.RL.environment.gym import StoneSoupEnv
$ from stonesoup.plugins.RL.scripts.train import main
```


## Integration guide <a id="integration-guide"></a>

**Contribution**

When making a pull requests in the StoneSoup GitHub, you must adhere to the
coding, documenting, and testing practices that standardise the StoneSoup
repository. This ensures that there is consistency between codeblocks that have
been contributed by different people. This in turn makes the code easier to read
and comprehend for other contributors.

**Requirements**

Make sure you run the following lines of code to install relevant code for the integration pipeline:
```sh
$ apt-get install python3-sphinx
$ python3 -m pip install coverage
$ pip install -U pytest
```
**Code style**

There are several rules related to the style of the code used in the StoneSoup repository. Following these rules will maintain code consistency. These rules are as follows:

* Use clear naming for variables, objects, etc. rather than mathematical short hand (e.g. measurement_matrix rather than H)
* Use standard Python errors and warnings
* At least one release cycle with deprecated warnings before removing/changing interface (excluding alpha/beta releases)
* No strict typing, exploit duck typing
* Each object has one role, modular sub-components
* Embrace abstract base classes
* Allowing inhomogeneity at the lower levels of the inheritance hierarchy
* New components should subclass from base types for each (e.g. Updater) where possible
* Use of Python abstract classes to have familiar behaviour to existing Python objects
* Avoid strict encapsulation (use of __methods) but use non-strict encapsulation where needed (_method).
* All code should follow PEP 8. Flake8 can be used to check for issues. If necessary, in-line ignores can be added.


**Documentation**

In Stone Soup, NumPy Doc style documentation is used, with documentation generated with Sphinx. It must be provided for all public interfaces, and should also be provided for private interfaces.

Where applicable, documentation should include reference to the paper and mathematical description. For new functionality provide Sphinx-Gallery example which demonstrate use case.

To build the documentation, use the following:
```sh
$ sphinx-build -W docs/source docs/build
```
You should find index.html file present in the docs/build/ directory.

The pages built by Sphinx-Gallery will be cached to speed up building process, but can be cleared to ensure that all examples build, using the following command:
```sh
$ git clean -xf docs/source/auto_*
```

**Tests**

PyTest is used for testing in Stone Soup. As much effort should be put into developing tests as the code. Tests should be provide to test functionality and also ensuring exceptions are raised or managed appropriately. Tests can be run (including Flake8) with the following:

```sh
$ pytest --flake8 stonesoup
```

Code coverage should also be checked, aiming for 100% coverage (noting coverage alone is not a measure of test quality), using Coverage.py. This can be run with the following (along with Flake8):

```sh
$ pytest --cov=stonesoup --cov-report=html --flake8 stonesoup
```

This will produce a report in htmlcov directory.

**Step-by-step**

Submissions to the Stone Soup GitHub are done via Pull Requests. The GitHub Flow development approach is used, consisting of the following steps:
1) Create a new branch on the StoneSoup Git Repository (make sure you have push access.) This can be done by visiting https://github.com/dstl/Stone-Soup, clicking 'Branches', clicking 'New branch', giving the branch a name, and setting the source as the branch you wish to build upon, usually the default branch 'main'.

2) Once this is done you can clone the new branch into a local repository on your computer by running the following command in your terminal (make sure that the current directory is where you want to have your local repository):

```sh
$ git clone -b <branch-name> git@github.com:decisionLabLtd/dstl-mast.git
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where \<branch-name> is substituted for the name you gave your remote branch.

3) Once this is done you will have a cloned local repository of the remote branch you created. You can now begin to make additions and changes to the code and file structure where required.

4) Following these changes, it will be worth running some local checks and tests to make sure the code meets the expected standard. If any new interfaces have been created, you must add them to their corresponding reStructuredText file (.rst) within the docs/source folder of the repository. These files are used by Sphinx to generate documentation. You can generate and store this documentation after updating the rst files by running the following command:

```sh
$ sphinx-build -W docs/source docs/build
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;You can then inspect the results in docs/build/index.html. You should find any new interfaces present in the documentation. If not then double check that the .rst files have been written correctly. You can use existing interfaces as examples.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Following that, you can run some tests using pytest. The following line of code will test the code to make sure it adheres to the parameters of the .flake8 file, as well as check the coverage of the code (aiming for 100%):

```sh
$ pytest --cov=stonesoup --cov-report=html --flake8 stonesoup
```



5) After making the desired changes, your next action will be to stage, commit, and push the files. This can all be done with a few lines of code in the terminal of the local repository. Firstly, you can check to see what files have been changed by running:

```sh
$ git status
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This will show what files have pending changes. You can promote these pending changes into the staging area by using the git add command. While you can run the command for each file changed, it is easier to simply run the following from the local repository root:

```sh
$ git add .
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; After that you can run a pre-commit to make sure that the code follows expected standards, including the flake8 once again. You can run pre-commit using the following line of code in the terminal:
```sh
pre-commit run --all-files
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; If any issues are highlighted, you can make the changes, and use git add once again to stage them.


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;After this is done you can then commit the staged changes to your repository. It is good practice to include a message stating what changes you made. The command for this is as follows:

```sh
$ git commit -m "Your message here"
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Once this is done you can push to the remote repository to add your changes to your branch on github. The command will be:

```sh
$ git push origin <branch-name>
```

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Where \<branch-name> is replaced with the name of your branch.


6) Now that your changes have been pushed to your branch of the Stone Soup repository, you next have to make a pull request to the branch you want to merge with (most likely main). To do this you must go to the Stone Soup GitHub page (https://github.com/dstl/Stone-Soup/), and click 'Compare & pull request' in the new yellow banner at the top of the page. Use the base branch dropdown menu to select the branch you want to merge the changes into. Use the compare branch dropdown menu to select the branch you made the changes in. Type a title and description explaining the changes you made. Click 'Create Pull Request'.

7) Finally, keep an eye on the pull request for both user reviews as well as results returned from CircleCI providing details on test results and documentation which it will have run automatically. You will also see an automated coverage review from CodeVec. After looking at this data, if any issues are flagged, make the relevant changes by starting from step 3 again. Once the code is satisfactory and is approved by the Stone Soup GitHub admins, the code will be merged and integrated with Stone Soup.
