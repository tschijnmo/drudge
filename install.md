---
title: Install
layout: page
---

# Execution by Docker

Due to the relatively complex dependencies on things like a new C++ compiler,
Python, and Apache Spark, Docker is recommended for users of the
drudge/gristmill stack.  On Linux platforms, no performance overhead is
expected.

All Docker images are built from the Git repository
[drudge-docker](https://github.com/tschijnmo/drudge-docker), and the built
images are all pushed into the Docker repository
[tschijnmo/drudge](https://cloud.docker.com/repository/registry-1.docker.io/tschijnmo/drudge).
To pull the latest image for drudge and gristmill,

```
docker pull tschijnmo/drudge:drudge
docker pull tschijnmo/drudge:gristmill
```

can do the job, with the gristmill step able to be omitted if gristmill is not
needed.  The initial pulling might take a few GBs of data.  However, later
updates are expected to be relatively small.  The docker images have got all
dependencies, as well as the latest drudge/gristmill installed.

For both images, the directory `/home/work` is for holding the working files
for a job.  For instance, if we have a script `script.py` in the current
working directory, running

```
docker run -it --rm -v $PWD:/home/work tschijnmo/drudge:gristmill
```

will launch a container with all the drudge/gristmill stack, with the current
working directory on the host machine mounted at `/home/work`.  Then in a
interactive shell in this container, the script can be executed by `python3
script.py`.  All outputs written to the `/home/work` directory will be directly
visible on the host machine.  If no interactive shell is desired,

```
docker run --rm -v $PWD:/home/work tschijnmo/drudge:gristmill python3 script.py
```

can also execute the script directly.


# Downloads and installation (Native)

For development, the drudge stack can also be downloaded, compiled, and
installed from source.  For most non-developmental users, execution by Docker
is recommended.

## Dependencies

In order to fully take advantage of the latest technology, the drudge/gristmill
stack requires Python at least 3.6, and Apache Spark at least 2.2 is need.  To
compile the binary components, a C++ compiler with good C++14 support is
required.  Clang++ later than 3.9 and g++ later than 6.3 is known to work.


## Downloads

All components of the drudge/gristmill stack are hosted on Github.  The
symbolic drudge code is hosted at
[tschijnmo/drudge](https://github.com/tschijnmo/drudge), which has the
[libcanon core](https://github.com/tschijnmo/libcanon) as a submodule for the
core algorithms for the canonicalization of combinatorial objects.

The code optimizer and generator gristmill is similarly hosted at
[tschijnmo/gristmill](https://github.com/tschijnmo/gristmill), which has the
submodules of

* `fbitset` at [tschijnmo/fbitset](https://github.com/tschijnmo/fbitset), for a
  highly-optimized bitmap container for the combinatorial algorithms in
  gristmill,
* `libparenth` at
  [tschijnmo/libparenth](https://github.com/tschijnmo/libparenth), for the core
  algorithm to find an optimal execution plan for tensor contractions by
  parenthesization, and
* `cpypp` at [tschijnmo/cpypp](https://github.com/tschijnmo/cpypp), for
  wrapping core C++ native modules for Python with ease.

As a result, to clone the repositories, `--recurse-submodules` is recommended.

## Compilation and installation

By `setuptools`, inside the root directory of the source tree of drudge or
gristmill, the compilation and installation can simply be

```
python3 setup.py build
python3 setup.py install
```

