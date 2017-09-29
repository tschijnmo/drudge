---
title: Install
layout: page
---

# Downloads and installation


## Dependencies

In order to fully take advantage of the latest technology, the drudge/gristmill
stack requires Python at least 3.6, and Spark at least 2.2 is need.  To compile
the binary components, a C++ compiler with good C++14 support is required.
Clang++ later than 3.9 and g++ later than 6.3 is known to work.


## Downloads

All components of the drudge/gristmill stack are hosted on Github, in separate
repositories for the [libcanon core](https://github.com/tschijnmo/libcanon),
the symbolic [drudge](https://github.com/tschijnmo/drudge), and the code
optimizer and generator [gristmill](https://github.com/tschijnmo/gristmill).
Source code of all historical releases are available for download.


## Installation

By `setuptools`, inside the root directory of the source tree of drudge or
gristmill, the installation can simply be

```
python3 setup.py build
python3 setup.py install
```

Direct installation from `pip` will be supported near future.

