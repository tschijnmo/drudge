import os

def FlagsForFile(filename, **kwargs):

    flags = ['-std=c++14', '-I/usr/local/include']

    proj_root = os.path.dirname(os.path.abspath(__file__))
    libcanon_include = ''.join(['-I', proj_root, '/deps/libcanon/include'])
    proj_include = ''.join(['-I', proj_root, '/drudge'])
    flags.extend([libcanon_include, proj_include])

    return {'flags': flags}
