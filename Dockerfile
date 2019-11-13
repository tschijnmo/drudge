# A simple docker image for Drudge

FROM tschijnmo/drudge:base

COPY . drudge

RUN set -ex; \
        cd drudge; \
        python3 setup.py build; \
        python3 setup.py install;

# For the convenience for running jobs.
RUN mkdir /home/work
WORKDIR /home/work
