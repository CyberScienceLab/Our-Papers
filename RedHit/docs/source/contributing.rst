Contributing
============

Getting ready
-------------

``garak``'s codebase is managed using github.
The best and only way to contribute to ``garak`` is to start by getting a copy of the source code.
You can use github's fork function to do this, which makes a copy of the ``garak`` codebase under your github account.
In there, you can branch, edit code and build.
Once you're done, make a pull request to the main repo and describe what you've done -- and we'll take it from there!

Checking your contribution is within scope
------------------------------------------

``garak`` is a security toolkit rather than a content safety or bias toolkit.
The project scope relates primarily to LLM & dialog system security.
This is a huge area, and you can get an idea of the kind of contributions that are in scope from our `FAQ <https://github.com/NVIDIA/garak/blob/main/FAQ.md>_` and our `Github issues <https://github.com/NVIDIA/garak/issues>`_ page.


Connecting with the ``garak`` team & community
----------------------------------------------

If you're going to contribute, it's a really good idea to reach out, so you have a source of help nearby, and so that we can make sure your valuable coding time is spent efficiently as a contributor.
There are a number of ways you can reach out to us:

* GitHub discussions: `<https://github.com/NVIDIA/garak/discussions>`_
* Twitter: `<https://twitter.com/garak_llm>`_
* Discord: `<https://discord.gg/uVch4puUCs>`_

We'd love to help, and we're always interested to hear how you're using garak.


Checklist for contributing
--------------------------

1. Set up a `Github <https://github.com/>`_ account, if you don't have one already. We develop in the open and the public repository is the authoritative one.
1. Fork the ``garak`` repository - `<https://github.com/NVIDIA/garak/fork>`_
1. Work out what you're doing. If it's from a good first issue (`see the list <https://github.com/NVIDIA/garak/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22>`_), drop a note on that issue so that we know you're working on it, and so that nobody else also starts working on it.
1. Before you code anything: create a new branch for your work, e.g. ``git checkout -b feature/spicy_probe``
1. Check out the rest of this page which includes links to detailed step-by-step guides to developing garak plugins
1. Code!
1. Run ``black`` on your code, so that it's well-formatted. Our github commit hook can refuse to accept ``black``-passing code.
1. Write your own tests - these are a requirement for merging!
1. When you're done, send a pull request. Github has big buttons for this and there's a template for you to fill in.
1. We'll discuss the code together with you, tune it up, and hopefully merge it in, maybe with some edits!
1. Now you're an official ``garak`` contributor, and will be permanently recognized in the project credits from the next official  release. Thank you!



Code structure
--------------

We have a page describing the :doc:`top-level concepts in garak <basic>`. 
Rather than repeat that, take a look, so you have an idea about the code base!

Developing your own plugins
---------------------------

Plugins are generators, probes, detectors, buffs, harnesses, and evaluators. Each category of plugin gets its own directory in the source tree. The first four categories are where most of the new functionality is.

The recipe for writing a new plugin or plugin class isn't outlandish:

* Only start a new module if none of the current modules could fit
* Take a look at how other plugins do it
   * For an example Generator, check out :class:`garak.probes.replicate`
   * For an example Probe, check out :class:`garak.probes.malwaregen`
   * For an example Detector, check out :class:`garak.detectors.toxicity` or :class:`garak.detectors.specialwords`
   * For an example Buff, check out :class:`garak.buffs.lowercase`
* Start a new module inheriting from one of the base classes, e.g. :class:`garak.probes.base.Probe`
* Override as little as possible.


Guides to writing plugins
-------------------------

Here are our tutorials on plugin writing:

* :doc:`Building a garak generator <contributing.generator>` -- step-by-step guide to building an interface for a real API-based model service
* :doc:`Building a garak probe <contributing.probe>` -- A guide to writing your own custom probes


Describing your code changes
----------------------------

Commit messages
~~~~~~~~~~~~~~~

Commit messages should describe what is changed in the commit. Try to keep one "theme" per commit. We read commit messages to work out what the intent of the commit is. We're all trying to save time here, and clear commit messages that include context can be a great time saver. Check out this guide to writing `commit messages <https://www.freecodecamp.org/news/how-to-write-better-git-commit-messages/>`_.

Pull requests
~~~~~~~~~~~~~
When you're ready, send a pull request. Include as much context as possible here. It should be clear why the PR is a good idea, what it adds, how it works, where the code/resources come from if you didn't create them yourself.

Review
~~~~~~
We review almost all pull requests, and we'll almost certainly chat with you about the code here. Please take this as a positive sign - we want to understand what's happening in the code. If you can, please also be reasonably responsive during code review; it's hard for us to merge code if we don't understand it or it does unusual things, and we can't contact the people who wrote it.


Testing
-------

Testing during development
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can test your code in a few ways:

* Start an interactive Python session
   * Instantiate the plugin, e.g. ``import garak._plugins`` then ``probe = garak._plugins.load_plugin("garak.probes.mymodule.MyProbe")``
   * Check out that the values and methods work as you'd expect
* Get ``garak`` to list all the plugins of the type you're writing, with ``--list_probes``, ``--list_detectors``, or ``--list_generators``: ```python3 -m garak --list_probes``
* Run a scan with test plugins
   * For probes, try a blank generator and always.Pass detector: ``python3 -m garak -m test.Blank -p mymodule -d always.Pass``
   * For detectors, try a blank generator and a blank probe: ``python3 -m garak -m test.Blank -p test.Blank -d mymodule``
   * For generators, try a blank probe and always.Pass detector: ``python3 -m garak -m mymodule -p test.Blank -d always.Pass``


garak supports pytest tests in garak/tests. You can run these with ``python -m pytest tests/`` from the root directory.
All the tests should pass for any code there's a pull request for, and all tests must pass in any PR before it can be merged.

Testing before sending a pull request
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Only code that passes the ``garak`` tests can be merged. Contributions must pass all tests.

Please write running tests to validate any new components or functions that you add.
They're pretty straightforward - you can look at the existing code in `tests` to get an idea of how to write these.
