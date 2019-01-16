Installation
============

Prerequisites
-------------

Python
~~~~~~
.. note::
    At the time of writting this document, the target Python versions for Stone-Soup are Python 3.5, Python 3.6 and Python 3.7. 

    To ensure that any Stone-Soup installation remains compatible for longer, it is advisable that the latest compatible version of Python is used, when possible.

Stone-Soup is a Python-based framework and thus a working Python installation is a mandatory requirement for one to work with and develop in Stone-Soup. 

Installation
************

1. Go to `<https://www.python.org/downloads/>`_
2. Search for and download the latest Python 3.7.x version. Alternatively, simply click on the yellow "Download Python 3.7.x" button 
3. Once you execute the downloaded ```python-3.7.x.exe``` installation file, on the window that opens **check the "Add Python 3.7 to PATH" checkbox** and click the "Install Now" option.
4. To confirm that Python 3 is correctly installed and configured, open a Windows Powershell/Command Promp window, type ```python -V``` and press Enter. This should print out something similar to ```Python 3.7.2```.
5. If instead you get a ```‘python’ is not recognized as an internal or external command``` error, it is very likely that you forgot to check the "Add Python 3.7 to PATH" checkbox (as instructed above) during the installation process. In this case simply uninstall and reinstall Python as per the above instructions.

.. note::
    It is worth noting that, aside from the official CPython distribution, there exist a number of other Python distributions that one can use (e.g. `Anaconda <https://www.anaconda.com/download/>`_, `Intel <https://software.intel.com/en-us/distribution-for-python>`_, etc.), each providing benefits in specific use-cases. 

    For the sake of ease and speed of installation, in this tutorial we will be installing the official Python distribution. However, any distribution of a Stone-Soup compatible Python version should work just as well. 



Git
~~~
The Stone-Soup code repository is hosted on `GitHub <https://github.com/dstl/Stone-Soup>`_ and uses `Git <https://git-scm.com/>`_ for version control. 

One notable feature of Git, and one that is vital to Stone-Soup development, is its `branching model <https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging>`_. In plain terms, a branch can be viewed as a copy/snapshot of the main/baseline code, which is a branch itself (typically termed `master`), or even another branch.   
    
This allows developers (i.e. us) to seamlessly and concurrently create, develop and colaborate on seperate branches (copies) of the main code (or other branches), which can be merged (a.k.a. applied) at a later time, through an intuitive merging process. 

Installation
************
1. Go to `<https://git-scm.com/>`_ and download the latest version of Git.
2. Execute the downloaded installation file and advance the process until the "Select Components" window is shown. 
3. In this window, the default selections under the "Windows Explorer Integration" group will result in the options being added to the context window the appears when you right-click on/inside a given directory in the Windows File Explorer. However, as explained further down, in this tutorial we will be using a 3rd party Git GUI Client. Thus, these options become obsolete and can be unselected.
4. Progress through the rest of the installation process using the provided defauls settings (unless otherwise desireable) and begin the installation. 

Developement tools
------------------

The following tools will be utilised to assist in the effective and effortless developement and version control for Stone-Soup:

1. Text Editor/Integrated Developer Environment (IDE)
    - Unlike MATLAB, Python does not come bundled with an IDE. 
    - Thus, it is up to the developer to select and utilise an IDE of their choice.

2. Git GUI Client
    - By default, Git only comes with command-line tools (i.e. Git Bash), which, although powerfull, are generally not very intuitive. 
    - To solve this problem, Git GUI Clients provide an intuitive GUI that abstracts users from the undelying commands.  


Microsoft Visual Studio Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The text Editor/IDE selected for the purposes of this tutorial is `Microsoft Visual Studio Code <https://code.visualstudio.com/>`_ (a.k.a. VS Code). This is due to the specific set of Extensions that one can install in VS Code, that provide huge benefits in terms of developing code and ensuring compliance with Stone-Soup coding and documentation standards. 

.. _Installation:

Installation
************
1. Go to `<https://code.visualstudio.com/>`_ and download a version of VS Code for your OS.
2. Execute the downloaded installation file and advance the process until the "Select Additional Tasks" window is shown.
3. In this window, selecting the checkboxes under the "Other:" section will enable you to open specific files/directories in VS Code by simply right-clicking on them and selecting the "Open with Code" option that is added to the context menu. From experience, this has proven as a very convenient alternative to the process of opening the editor and following the standard ```File > Open...``` sequence. Thus it is advisable that you select these options. 
4. Proceed to the next step and click "Install" to begin the installation.

.. note::
    There exist various popular alternative IDEs, such as `PyCharm <https://www.jetbrains.com/pycharm/>`_, `Sublime Text <https://www.sublimetext.com/>`_ and `Visual Studio <https://visualstudio.microsoft.com/vs/>`_ (not to be mistaken with VS Code), which developers may also use if desireable, however the extensions and configuration process documented below does not apply to these editors.

Extensions
**********
There are two main extensions, which can be installed and configured to assist in ensuring compliance with Stone-Soup coding standards.

1. `Python (by Microsoft) <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`_:

 - This extension provides linting, debugging (multi-threaded, remote), Intellisense, code formatting, refactoring, unit tests, snippets, and more capabilities when editing Python files.
 - Most notably, when configured with (a) specific linter(s) (e.g. flake8, autopep8) and the appropriate settings (which we will discuss further down), the Extension provides:
   
   * Clear highlighting and feedback on any syntactic and/or `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ [#f1]_ errors.
   * Interractive debugging of Python scripts/libraries, similar to that present in MATLAB.
   * Auto-formatting (e.g. on File Close/Save) of code, when possible, to ensure PEP 8 style compliance.
   * Code suggestions and auto-complete functionality for classes, functions, variables, etc. 

2. `autoDocstring (by Nils Werner) <https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring>`_:

 - Stone-Soup uses `Sphinx <http://www.sphinx-doc.org/en/master/>`_ [#f2]_, combined with appropriatelly structured code comments, called *docstrings*, to generate documentation directly from code.
 - For the above to be achieved, Sphinx requires that certain conventions (see `Sphinx and reST <https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html>`_ and `Sphinx Numpy\\Scipy docstring format <https://docs.scipy.org/doc/numpy-1.15.0/docs/howto_document.html>`_ for more details) are followed when writing such docstrings.
 - Once a Stone-Soup class/function has been written, this extension provides automatic generation of partially filled docstrings, based on the contents of the target class/function.

Installing extensions
+++++++++++++++++++++

In VS Code, extensions can be installed as follows:

1. Click on the "Extensions" icon found on the left vertical toolbar of VS code to open the Extensions Marketplace.
2. Search for and select the desired extension.
3. Click on the "Install" button at the top of the extension page.
4. Typically, a reload of the editor is required once an extension has been installed. This can be one by clicking on the "Reload" button that appears in place of the "Install" button clicked before.

.. [#f1] Stone-Soup uses the PEP 8 Code Styling guidelines. PEP 8 compliance is mandatory for any code submitted for use in Stone-Soup to be accepted for merging. 
.. [#f2] Stone-Soup uses Sphinx to generate documentation. A standard Stone-Soup requirement for contributing code is that it is appropriately documented.

Configuration
*************
Once installed, extensions must be configured to operate in the desired manner. This can be done as follows:

1. Go to ```File > Preferences > Settings```. This opens up a GUI based Settings editor. 
2. VS Code settings are stored in JSON files. In the top-right corner of the Settings tab, click on the button depicting a ```{}``` symbol to view the underlying JSON files.
3. On the right half of tab that opens up, make sure that the "USER SETTINGS" tab is selected.
4. Copy and paste the following settings in the provided editor window:

    .. code::

        {
            "[python]": {
                "editor.formatOnSave": True
            },
            "python.linting.flake8Enabled": true,
            "python.linting.pylintEnabled": false,
            "autoDocstring.docstringFormat": "numpy"
        }

5. Save the changes using the key combination ```Ctrl+S``` or by going to ```File > Save```.

Below is an outline of the effect of the applied settings:

- ``"editor.formatOnSave": true`` - This setting will enable the editor to apply auto-formatting (using ``autopep8``, by default) when saving Python files, to ensure PEP 8 compliance.
    
    To visualise the effect of this optiion, create a new "\*.py" file and paste the following code as provided (awkwardly spaced):

    .. code::
        
        a      =                4
        b=a-4
    
    Then Save the file and notice how the code is automatically formatted to: 

    .. code::
    
        a = 4
        b = a-4

    If the above doesn't go to plan and instead a "Formatter autopep8 is not installed. Install?" editor error appears, click "Yes" in the error pop up window to proceed to install ``autopep8``. Once this is done, retry the above process.

- ``"python.linting.flake8Enabled": true`` - Instructs the Python extension to use ``flake8`` for linting (as per Stone-Soup conventions) 
- ``"python.linting.pylintEnabled": false`` - Instructs the Python extension NOT to use ``pylint`` for linting
- ``"autoDocstring.docstringFormat": "numpy"`` - Instructs the autoDocstring extension to generate docstrings using the `Sphinx Numpy\\Scipy docstring format <https://docs.scipy.org/doc/numpy-1.15.0/docs/howto_document.html>`_ (as per Stone-Soup conventions) 

SmartGit
~~~~~~~~

The Git GUI Client selected for the purposes of this tutorial is `SmartGit <https://www.syntevo.com/smartgit//>`_. SmartGit is a graphical Git client with support for SVN and Pull Requests for GitHub. 

.. note::

    SmartGit can be used free of charge by Open Source developers and academics [#f3]_. This means that only work that relates to the Open Source version of Stone-Soup can be performed using SmartGit.

    A fully free, but less functional alternative is `GitHub Desktop <https://desktop.github.com/>`_

    .. [#f3] `<https://www.syntevo.com/smartgit/purchase/>`_
    
Installation
************
1. Go to `<https://www.syntevo.com/smartgit/download/>`_ and download a version of SmartGit for Windows.
2. Extract the contents of the downloaded folder and proceed to execute the contained "\*.exe" installation file.
3. Progress through the installation process using the provided defauls settings (unless otherwise desireable) and begin the installation.
4. Once the installation completed, a setup window will be opened.
5. Check the terms and conditions checkbox and **select the "Non-commercial use ony (most features, no support)" option**.
6. On the next window, set up your `Git Credentials <https://git-scm.com/docs/gitcredentials>`_ by inserting a User Name of you choice and the email address of your GitHub account (that has access to Stone-Soup).
7. Progress through the rest of the setup steps using the default settings.
8. SmartGit should open automatically once the setup is finished.

Stone-Soup
----------


Downloading\\Cloning
~~~~~~~~~~~~~~~~~~~~

To download a copy of the main/baseline code of Stone-Soup, we proceed by cloning the Stone-Soup ``master`` Git branch. Among other ways, this can either be done through SmartGit (GUI), if installed, or through Git Bash (command-line). Below we show both ways, mainly to demonstrate how the two relate to each other.

Using SmartGit
**************
1. Open SmartGit.
2. Go to ```Repository > Clone```
3. In the window that opens up, select "Remote Git or SVN repository" (if not already selected), paste the ``https://github.com/dstl/Stone-Soup`` URL in the "Repository URL" field and click "Next".
4. In the next window you can select the branch that you wish to clone. The default selection should be ``master``, which, as mentioned before, is the one we wish to clone. Click "Next".
5. In the 3rd (and final) window, provide a path on your machine where the repository should be downloaded and click "Finish".
6. You can now open the residing directory of Stone-Soup by simply right-clicking on "Stone-Soup (master)" under the Repositories sub-window (typically on the left) and selecting "Open in Explorer". 

Using Git Bash
**************
1. Open Git Bash.
2. If not done previously, set up your Git credentials, by running the following commands:
    
    .. code::

        git config --global user.name myusername
        git config --global user.email myemail

    where ``myusername`` is any username of your choice and ``myemail`` should be the email associated to your GitHub account, that has access to the Stone Soup repository.
3. ``cd`` into the directory where you wish the Stone Soup repository to be placed, e.g. assuming the directory already exists:

    ..  code::

        cd "C:/Users/sglvladi/Documents/Repositories/"

4. Clone a copy of the Stone-Soup `master` branch:

    .. code:: 

        git clone -b master https://github.com/dstl/Stone-Soup

5. Once this is done, a "Stone-Soup" directory will be created inside the current working directory, which contains the newly cloned branch. 

Installation
~~~~~~~~~~~~
1. Open the Stone-Soup repository folder in VS Code. If you previously set the "Open with Code" options (as advised in Step 3 of the VS Code Installation_), this can be achieved by right-clicking the repository folder using File Explorer, and selecting the relevant option.
2. Go to ```View > Terminal``` to open the VS Code Terminal Window.
3. Ensure that the terminal points to the root repository folder (i.e. ``<SOME_PATH>/Stone-Soup``) and execute:

    .. code::

        python -m pip install .[dev]
    
    The ``.[dev]`` option installs all Python dependencies necessary for developement.

3. If the process terminates successfully, Stone-Soup and all of its dependencies will be installed on your computer.
4. To your Stone-Soup installation, run the following command in the Terminal:

    .. code:: 

        pytest .\stonesoup\
5. The above command runs through all unit test files in Stone-Soup, ensuring that all framework components behave as expected.
6. If the test process completes without any errors, this means that **Stone-Soup is installed and ready to use**.

