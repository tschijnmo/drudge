"""Utility to write simple reports in HTML format."""

import subprocess
import sys

from jinja2 import Environment, PackageLoader


class Report:
    """Simple report for output drudge results.

    This class helps to write symbolic results to files for batch processing
    jobs.  It is not recommended to be used directly.  Users should use the
    method provided by drudge class instead in ``with`` statements.
    """

    def __init__(self, filename: str, title):
        """Initialize the report object."""

        self._filename = filename

        filename_parts = filename.split('.')
        if len(filename_parts) < 2:
            raise ValueError(
                'Invalid filename, unable to determine type', filename
            )
        ext = filename_parts[-1].lower()
        if ext not in {'html', 'tex', 'pdf'}:
            raise ValueError('Invalid extension', ext, 'in', filename)
        self._ext = ext
        self._basename = '.'.join(filename_parts[:-1])

        self._sects = []
        self._ctx = {
            'title': title,
            'sects': self._sects
        }

    def add(self, title, content, description='', env='[', **kwargs):
        r"""Add a section to the result.

        Parameters
        ----------

        title

            The title of the equation.  It will be used as a section header.

        content

            The actual tensor or tensor definition to be printed.

        description

            A verbal description of the content.  It will be typeset before the
            actual equation as normal text.

        env

            The environment to put the equation in.  A value of ``'['`` will use
            ``\[`` and ``\]`` as the deliminator of the math environment.  Other
            values will be put inside the common ``\begin{}`` and ``\end{}``
            tags of LaTeX.

        kwargs

            All the rest of the keyword arguments are forwarded to the
            :py:meth:`Drudge.format_latex` method.

        Note
        ----

        **For long equations in LaTeX environments,** normally ``env='align'``
        and ``sep_lines=True`` can be set to allow each term to occupy separate
        lines, automatic page break will be inserted, or ``env='dmath'`` and
        ``sep_lines=False`` can be used to use ``breqn`` package to
        automatically flow the terms.

        """

        if env == '[':
            opening, closing = r'\[', r'\]'
        else:
            opening, closing = [
                r'\{}{{{}}}'.format(i, env)
                for i in ['begin', 'end']
            ]

        self._sects.append({
            'title': title,
            'description': description,
            'expr': content.latex(**kwargs),
            'opening': opening,
            'closing': closing
        })

    def write(self):
        """Write the report.

        Note that this method also closes the output file.
        """

        env = Environment(
            loader=PackageLoader('drudge'),
            lstrip_blocks=True, trim_blocks=True
        )

        if self._ext == 'html':
            templ_name = 'report.html'
            filename = self._filename
        elif self._ext == 'tex' or self._ext == 'pdf':
            templ_name = 'report.tex'
            filename = self._basename + '.tex'
        else:
            assert False

        templ = env.get_template(templ_name)

        with open(filename, 'w') as fp:
            templ.stream(self._ctx).dump(fp)

        if self._ext == 'pdf':
            stat = subprocess.run(
                ['pdflatex', filename],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Do not crash program only because LaTeX does not compile.
            if stat.returncode != 0:
                err_msg = 'pdflatex failed for {}.  Error: \n{}\n{}'.format(
                    filename, stat.stdout, stat.stderr
                )
                print(err_msg, file=sys.stderr)
