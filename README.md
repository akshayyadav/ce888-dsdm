
### Installation

* Required:
  - Python3 (for specific version refer to .python-version)
  - pip
  - Use of a python version manager like pyenv (or a virualenv) is highly recommended.

* Use pip and `requirements.txt` to install packages
    ```
    pip install -r requirements.txt --user
    ```

* If model serialisation(save/load) is giving errors try to use `requirements-freeze.txt` to install specific versions of all the packages available at the time of writing
    ```
    pip install -r requirements-freeze.txt --user
    ```
