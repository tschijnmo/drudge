import os
import subprocess

def FlagsForFile(filename, **kwargs):

    flags = ['-std=c++14', '-I/usr/local/include']

    proj_root = os.path.dirname(os.path.abspath(__file__))
    libcanon_include = ''.join(['-I', proj_root, '/deps/libcanon/include'])
    python_include = subprocess.run(
        ["pkg-config", '--cflags', 'python3'], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    proj_include = ''.join(['-I', proj_root, '/drudge'])
    flags.extend([libcanon_include, proj_include])

    return {'flags': flags}
