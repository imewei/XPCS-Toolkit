============
Contributing
============

Types of Contributions
----------------------

* Bug reports: https://github.com/imewei/XPCS-Toolkit/issues
* Feature requests: https://github.com/imewei/XPCS-Toolkit/issues
* Code contributions
* Documentation improvements

Development Setup
-----------------

.. code-block:: bash

   # Fork and clone
   git clone git@github.com:your_name_here/xpcs-toolkit.git
   cd xpcs-toolkit

   # Install
   pip install -e .[dev]
   make dev-setup

   # Create branch
   git checkout -b your-feature-name

   # Test changes
   make test
   make lint

   # Commit and push
   git commit -m "Description"
   git push origin your-feature-name

Pull Request Guidelines
-----------------------

* Include tests for new features
* Update documentation for changes
* Support Python 3.12 and 3.13
* Pass all quality checks

Code Standards
--------------

* Use ruff for linting/formatting
* Add type hints to functions
* Write tests for new code
* Add docstrings to public functions

Code of Conduct
---------------

Please note that this project is released with a `Contributor Code of Conduct`_.
By participating in this project you agree to abide by its terms.

.. _`Contributor Code of Conduct`: CODE_OF_CONDUCT.rst
