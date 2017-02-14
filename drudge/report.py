"""Utility to write simple reports in HTML format."""

from jinja2 import Environment, PackageLoader


class Report:
    """Simple report for output drudge results.

    This class helps to write symbolic results to HTML files for batch
    processing jobs.  It is not recommended to be used directly.  Users should
    use the method provided by drudge class instead in ``with`` statements.
    """

    def __init__(self, filename, title):
        """Initialize the report object."""

        self._filename = filename

        self._sects = []
        self._ctx = {
            'title': title,
            'sects': self._sects
        }

    def add(self, title, content, description='', sep_lines=True):
        """Add a section to the result."""
        self._sects.append({
            'title': title,
            'description': description,
            'expr': content.latex(sep_lines=sep_lines)
        })

    def write(self):
        """Write the report.

        Note that this method also closes the output file.
        """

        env = Environment(
            loader=PackageLoader('drudge'),
            lstrip_blocks=True, trim_blocks=True
        )
        templ = env.get_template('report.html')

        with open(self._filename, 'w') as fp:
            templ.stream(self._ctx).dump(fp)
